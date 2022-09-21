from ocpmodels.trainers.forces_trainer import ForcesTrainer
from ocpmodels.datasets.lmdb_dataset import data_list_collater
from ocpmodels.common.utils import setup_imports, setup_logging
from ocpmodels.common import distutils
import logging
import yaml
from ocpmodels.preprocessing import AtomsToGraphs
from ocpmodels.modules.loss import DDPLoss, L2MAELoss
import os
import torch
import copy
import torch.nn as nn
from finetuna.finetuner_utils.loss import (
    RelativeL2MAELoss,
    AtomwiseL2LossNoBatch,
)


class Trainer(ForcesTrainer):
    def __init__(self, config_yml=None, checkpoint=None, cutoff=6, max_neighbors=50):
        setup_imports()
        setup_logging()

        # Either the config path or the checkpoint path needs to be provided
        assert config_yml or checkpoint is not None

        if config_yml is not None:
            if isinstance(config_yml, str):
                config = yaml.safe_load(open(config_yml, "r"))

                if "includes" in config:
                    for include in config["includes"]:
                        # Change the path based on absolute path of config_yml
                        path = os.path.join(config_yml.split("configs")[0], include)
                        include_config = yaml.safe_load(open(path, "r"))
                        config.update(include_config)
            else:
                config = config_yml
        else:
            # Loads the config from the checkpoint directly
            config = torch.load(checkpoint, map_location=torch.device("cpu"))["config"]

            # Load the trainer based on the dataset used
            if config["task"]["dataset"] == "trajectory_lmdb":
                config["trainer"] = "forces"
            else:
                config["trainer"] = "energy"

            config["model_attributes"]["name"] = config.pop("model")
            config["model"] = config["model_attributes"]

        # Calculate the edge indices on the fly
        self.otf_graph = True
        config["model"]["otf_graph"] = self.otf_graph

        # Save config so obj can be transported over network (pkl)
        self.config = copy.deepcopy(config)
        self.config["checkpoint"] = checkpoint

        if "normalizer" not in config:
            del config["dataset"]["src"]
            config["normalizer"] = config["dataset"]

        identifier = ""
        if hasattr(config.get("logger", {}), "get"):
            identifier = config.get("logger", {}).get("identifier", "")

        super().__init__(
            task=config["task"],
            model=config["model"],
            dataset=None,
            optimizer=config["optim"],
            identifier=identifier,
            normalizer=config["normalizer"],
            slurm=config.get("slurm", {}),
            local_rank=config.get("local_rank", 0),
            logger=config.get("logger", None),
            print_every=config.get("print_every", 1),
            is_debug=config.get("is_debug", True),
            cpu=config.get("cpu", True),
        )

        # if loading a model with added blocks for training from the checkpoint, set strict loading to False
        if self.config["model"] in ["adapter_gemnet_t"]:
            self.model.load_state_dict.__func__.__defaults__ = (False,)

        # load checkpoint
        if checkpoint is not None:
            try:
                self.load_checkpoint(checkpoint)
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
            tags = atoms.get_tags()
            if atoms.constraints != []:
                tags[atoms.constraints[0].get_indices()] = 1
            else:
                tags = [1] * len(atoms)
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

    def train(self, disable_eval_tqdm=False):
        eval_every = self.config["optim"].get("eval_every", None)
        if eval_every is None:
            eval_every = len(self.train_loader)
        checkpoint_every = self.config["optim"].get("checkpoint_every", eval_every)
        primary_metric = self.config["task"].get(
            "primary_metric", self.evaluator.task_primary_metric[self.name]
        )
        self.best_val_metric = 1e9 if "mae" in primary_metric else -1.0
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

                if self.config["optim"]["optimizer"] == "LBFGS":

                    def closure():
                        self.optimizer.zero_grad()
                        with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                            out = self._forward(batch)
                            loss = self._compute_loss(out, batch)
                        loss.backward()
                        return loss

                    self.optimizer.step(closure)

                    self.optimizer.zero_grad()
                    with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                        out = self._forward(batch)
                        loss = self._compute_loss(out, batch)

                else:
                    # Forward, loss, backward.
                    with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                        out = self._forward(batch)
                        loss = self._compute_loss(out, batch)
                    loss = self.scaler.scale(loss) if self.scaler else loss
                    self._backward(loss)

                scale = self.scaler.get_scale() if self.scaler else 1.0

                # Compute metrics.
                self.metrics = self._compute_metrics(
                    out,
                    batch,
                    self.evaluator,
                    self.metrics,
                )
                self.metrics = self.evaluator.update(
                    "loss", loss.item() / scale, self.metrics
                )

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
                    and not self.is_hpo
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
                    if self.test_loader is not None:
                        test_metrics = self.validate(
                            split="test",
                            disable_tqdm=disable_eval_tqdm,
                        )
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
                        if self.is_hpo:
                            self.hpo_update(
                                self.epoch,
                                self.step,
                                self.metrics,
                                val_metrics,
                            )

                    if self.config["task"].get("eval_relaxations", False):
                        if "relax_dataset" not in self.config["task"]:
                            logging.warning(
                                "Cannot evaluate relaxations, relax_dataset not specified"
                            )
                        else:
                            self.run_relaxations()

                if self.config["optim"].get("print_loss_and_lr", False):
                    if self.step % eval_every == 0 or not self.config["optim"].get(
                        "print_only_on_eval", True
                    ):
                        if self.val_loader is not None:
                            print(
                                "epoch: "
                                + "{:.1f}".format(self.epoch)
                                + ", \tstep: "
                                + str(self.step)
                                + ", \tloss: "
                                + str(loss.detach().item())
                                + ", \tlr: "
                                + str(self.scheduler.get_lr())
                                + ", \tval: "
                                + str(val_metrics["loss"]["metric"])
                            )
                        else:
                            print(
                                "epoch: "
                                + "{:.1f}".format(self.epoch)
                                + ", \tstep: "
                                + str(self.step)
                                + ", \tloss: "
                                + str(loss.detach().item())
                                + ", \tlr: "
                                + str(self.scheduler.get_lr())
                            )

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

                break_below_lr = (
                    self.config["optim"].get("break_below_lr", None) is not None
                ) and (self.scheduler.get_lr() < self.config["optim"]["break_below_lr"])
                if break_below_lr:
                    break
            if break_below_lr:
                break

            torch.cuda.empty_cache()

            if checkpoint_every == -1:
                self.save(checkpoint_file="checkpoint.pt", training_state=True)

        self.train_dataset.close_db()
        if "val_dataset" in self.config:
            self.val_dataset.close_db()
        if "test_dataset" in self.config:
            self.test_dataset.close_db()

    def load_loss(self):
        self.loss_fn = {}
        self.loss_fn["energy"] = self.config["optim"].get("loss_energy", "mae")
        self.loss_fn["force"] = self.config["optim"].get("loss_force", "mae")
        for loss, loss_name in self.loss_fn.items():
            if loss_name in ["l1", "mae"]:
                self.loss_fn[loss] = nn.L1Loss()
            elif loss_name == "mse":
                self.loss_fn[loss] = nn.MSELoss()
            elif loss_name == "l2mae":
                self.loss_fn[loss] = L2MAELoss()
            elif loss_name == "rell2mae":
                self.loss_fn[loss] = RelativeL2MAELoss()
            elif loss_name == "atomwisel2":
                self.loss_fn[loss] = AtomwiseL2LossNoBatch()
            else:
                raise NotImplementedError(f"Unknown loss function name: {loss_name}")
            if distutils.initialized():
                self.loss_fn[loss] = DDPLoss(self.loss_fn[loss])
