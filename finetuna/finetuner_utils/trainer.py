from ocpmodels.trainers.ocp_trainer import OCPTrainer
from ocpmodels.datasets.lmdb_dataset import data_list_collater
from ocpmodels.common.utils import setup_imports, setup_logging, update_config
from ocpmodels.common import distutils
from ocpmodels.common.registry import registry
import logging
import yaml
from ocpmodels.preprocessing import AtomsToGraphs
from ocpmodels.modules.loss import DDPLoss, L2MAELoss
import os
import torch
import copy
import torch.nn as nn
import numpy as np
from finetuna.finetuner_utils.loss import (
    RelativeL2MAELoss,
    AtomwiseL2LossNoBatch,
)


class Trainer(OCPTrainer):
    def __init__(
        self, config_yml=None, checkpoint_path=None, cutoff=6, max_neighbors=50
    ):
        setup_imports()
        setup_logging()

        # Either the config path or the checkpoint path needs to be provided
        assert config_yml or checkpoint_path is not None

        checkpoint = None
        if config_yml is not None:
            if isinstance(config_yml, str):
                config, duplicates_warning, duplicates_error = load_config(config_yml)
                if len(duplicates_warning) > 0:
                    logging.warning(
                        f"Overwritten config parameters from included configs "
                        f"(non-included parameters take precedence): {duplicates_warning}"
                    )
                if len(duplicates_error) > 0:
                    raise ValueError(
                        f"Conflicting (duplicate) parameters in simultaneously "
                        f"included configs: {duplicates_error}"
                    )
            else:
                config = config_yml

            # Only keeps the train data that might have normalizer values
            # if isinstance(config["dataset"], list):
            #     config["dataset"] = config["dataset"][0]
            # elif isinstance(config["dataset"], dict):
            #     config["dataset"] = config["dataset"].get("train", None)
        else:
            # Loads the config from the checkpoint directly (always on CPU).
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            config = checkpoint["config"]

        # if trainer is not None:
        #     config["trainer"] = trainer
        # else:
        config["trainer"] = config.get("trainer", "ocp")

        if "model_attributes" in config:
            config["model_attributes"]["name"] = config.pop("model")
            config["model"] = config["model_attributes"]

        # for checkpoints with relaxation datasets defined, remove to avoid
        # unnecesarily trying to load that dataset
        if "relax_dataset" in config["task"]:
            del config["task"]["relax_dataset"]

        # Calculate the edge indices on the fly
        self.otf_graph = True
        config["model"]["otf_graph"] = self.otf_graph
        # Save config so obj can be transported over network (pkl)
        config = update_config(config)
        self.config = copy.deepcopy(config)
        self.config["checkpoint"] = checkpoint_path
        del config["dataset"]["src"]
        super().__init__(
            task=config["task"],
            model=config["model"],
            dataset=[config["dataset"]],
            outputs=config["outputs"],
            loss_fns=config["loss_fns"],
            eval_metrics=config["eval_metrics"],
            optimizer=config["optim"],
            identifier="",
            slurm=config.get("slurm", {}),
            local_rank=config.get("local_rank", 0),
            is_debug=config.get("is_debug", True),
            cpu=config.get("cpu", True),
            amp=config.get("amp", False),
        )

        # if loading a model with added blocks for training from the checkpoint, set strict loading to False
        if self.config["model"] in ["adapter_gemnet_t", "adapter_gemnet_oc"]:
            self.model.load_state_dict.__func__.__defaults__ = (False,)

        # load checkpoint
        if checkpoint_path is not None:
            try:
                self.load_checkpoint(checkpoint_path)
            except NotImplementedError:
                logging.warning("Unable to load checkpoint!")

        self.a2g_predict = AtomsToGraphs(
            max_neigh=max_neighbors,
            radius=cutoff,
            r_energy=False,
            r_forces=False,
            r_distances=False,
            r_edges=False,
        )

        self.a2g_train = AtomsToGraphs(
            max_neigh=max_neighbors,
            radius=cutoff,
            r_energy=True,
            r_forces=True,
            r_distances=True,
            r_edges=False,
        )

    def a2g_convert(self, atoms, train: bool):
        if "tags" not in atoms.arrays:
            tags = np.array([1] * len(atoms))
            if atoms.constraints != []:
                tags[atoms.constraints[0].get_indices()] = 0

            atoms.arrays["tags"] = tags

        if train:
            data_object = self.a2g_train.convert(atoms)
        else:
            data_object = self.a2g_predict.convert(atoms)

        return data_object

    def get_atoms_prediction(self, atoms):
        data_object = self.a2g_convert(atoms, False)
        batch = data_list_collater([data_object], self.otf_graph)
        predictions = self.predict(
            data_loader=batch, per_image=False, results_file=None, disable_tqdm=True
        )
        energy = predictions["energy"].item()
        forces = predictions["forces"].cpu().numpy()
        return energy, forces

    def save(
        self,
        metrics=None,
        checkpoint_file="checkpoint.pt",
        training_state=True,
    ):
        """
        Overwriting save file to make sure that checkpoints that have been trained with finetuner calc retain normalizer dictionary in the right location
        (by default 'Normalizers' was being saved by OCP, but when loading trained checkpoints it also expects the information to be saved in 'Normalizer')
        """
        checkpoint_path = super().save(
            metrics=metrics,
            checkpoint_file=checkpoint_file,
            training_state=training_state,
        )
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path)
            checkpoint["config"]["normalizer"] = self.normalizer
            torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def _compute_loss(self, out, batch):
        batch_size = batch.natoms.numel()
        fixed = batch.fixed
        mask = fixed == 0

        loss = []
        for loss_fn in self.loss_fns:
            target_name, loss_info = loss_fn

            target = batch[target_name]
            pred = out[target_name]
            natoms = batch.natoms
            natoms = torch.repeat_interleave(natoms, natoms)

            if (
                self.output_targets[target_name]["level"] == "atom"
                and self.output_targets[target_name]["train_on_free_atoms"]
            ):
                target = target[mask]
                pred = pred[mask]
                natoms = natoms[mask]

            num_atoms_in_batch = natoms.numel()
            if self.normalizers.get(target_name, False):
                target = self.normalizers[target_name].norm(target)

            ### reshape accordingly: num_atoms_in_batch, -1 or num_systems_in_batch, -1
            if self.output_targets[target_name]["level"] == "atom":
                target = target.view(num_atoms_in_batch, -1)
            else:
                target = target.view(batch_size, -1)

            mult = loss_info["coefficient"]
            loss.append(
                mult
                * loss_info["fn"](
                    pred,
                    target,
                    natoms=natoms,
                    batch_size=batch_size,
                )
            )

        # Sanity check to make sure the compute graph is correct.
        for lc in loss:
            assert hasattr(lc, "grad_fn")

        loss = sum(loss)
        return loss

    def train(self, disable_eval_tqdm: bool = False) -> None:
        # ensure_fitted(self._unwrapped_model, warn=True)

        eval_every = self.config["optim"].get("eval_every", len(self.train_loader))
        checkpoint_every = self.config["optim"].get("checkpoint_every", eval_every)
        primary_metric = self.evaluation_metrics.get(
            "primary_metric", self.evaluator.task_primary_metric[self.name]
        )
        if not hasattr(self, "primary_metric") or self.primary_metric != primary_metric:
            self.best_val_metric = 1e9 if "mae" in primary_metric else -1.0
        else:
            primary_metric = self.primary_metric
        self.metrics = {}

        # Calculate start_epoch from step instead of loading the epoch number
        # to prevent inconsistencies due to different batch size in checkpoint.
        start_epoch = self.step // len(self.train_loader)

        for epoch_int in range(start_epoch, self.config["optim"]["max_epochs"]):
            self.train_sampler.set_epoch(epoch_int)
            skip_steps = self.step % len(self.train_loader)
            train_loader_iter = iter(self.train_loader)

            for i in range(skip_steps, len(self.train_loader)):
                self.epoch = epoch_int + (i + 1) / len(self.train_loader)
                self.step = epoch_int * len(self.train_loader) + i + 1
                self.model.train()

                # Get a batch.
                batch = next(train_loader_iter)

                # Forward, loss, backward.
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    out = self._forward(batch)
                    loss = self._compute_loss(out, batch)

                # Compute metrics.
                self.metrics = self._compute_metrics(
                    out,
                    batch,
                    self.evaluator,
                    self.metrics,
                )
                self.metrics = self.evaluator.update("loss", loss.item(), self.metrics)

                loss = self.scaler.scale(loss) if self.scaler else loss
                self._backward(loss)

                # Log metrics.
                log_dict = {k: self.metrics[k]["metric"] for k in self.metrics}
                log_dict.update(
                    {
                        "lr": self.scheduler.get_lr(),
                        "epoch": self.epoch,
                        "step": self.step,
                    }
                )
                if (
                    self.step % self.config["cmd"]["print_every"] == 0
                    and distutils.is_master()
                ):
                    log_str = ["{}: {:.2e}".format(k, v) for k, v in log_dict.items()]
                    logging.info(", ".join(log_str))
                    self.metrics = {}

                if self.logger is not None:
                    self.logger.log(
                        log_dict,
                        step=self.step,
                        split="train",
                    )

                if checkpoint_every != -1 and self.step % checkpoint_every == 0:
                    self.save(checkpoint_file="checkpoint.pt", training_state=True)

                # Evaluate on val set every `eval_every` iterations.
                if self.step % eval_every == 0:
                    if self.val_loader is not None:
                        val_metrics = self.validate(
                            split="val",
                            disable_tqdm=disable_eval_tqdm,
                        )
                        self.update_best(
                            primary_metric,
                            val_metrics,
                            disable_eval_tqdm=disable_eval_tqdm,
                        )

                    if self.config["task"].get("eval_relaxations", False):
                        if "relax_dataset" not in self.config["task"]:
                            logging.warning(
                                "Cannot evaluate relaxations, relax_dataset not specified"
                            )
                        else:
                            self.run_relaxations()

                if self.scheduler.scheduler_type == "ReduceLROnPlateau":
                    if (
                        self.step % eval_every == 0
                        and self.config["optim"].get("scheduler_loss", None) == "train"
                    ):
                        self.scheduler.step(
                            metrics=loss.detach().item(),
                        )
                    elif self.step % eval_every == 0 and self.val_loader is not None:
                        self.scheduler.step(
                            metrics=val_metrics[primary_metric]["metric"],
                        )
                else:
                    self.scheduler.step()

            torch.cuda.empty_cache()

            if checkpoint_every == -1:
                self.save(checkpoint_file="checkpoint.pt", training_state=True)

        self.train_dataset.close_db()
        if self.config.get("val_dataset", False):
            self.val_dataset.close_db()
        if self.config.get("test_dataset", False):
            self.test_dataset.close_db()
