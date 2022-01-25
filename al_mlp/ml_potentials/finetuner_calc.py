from ase.calculators.calculator import all_changes
from ase.atoms import Atoms
from al_mlp.ml_potentials.ml_potential_calc import MLPCalc
from torch.utils.data import Dataset
from ocpmodels.preprocessing import AtomsToGraphs
import sys, os
import yaml
from al_mlp.job_creator import merge_dict
import copy
import time
import torch
from ocpmodels.trainers.forces_trainer import ForcesTrainer
from ocpmodels.datasets.trajectory_lmdb import data_list_collater
from ocpmodels.common.utils import setup_imports, setup_logging
from ocpmodels.common import distutils
import logging
import numpy as np


class FinetunerCalc(MLPCalc):
    """
    Open Catalyst Project Finetuner/Transfer learning calculator.
    This class serves as a parent class for calculators that want to use ensembling on one of the ocp models for finetuning.
    On its own this class instantiates one ocp model with a simulated uncertainty.

    By default simply ticks up a counter to simulate std uncertainty metric.
    Ensemble child class calcs implementing uncertainty should overwrite train_ocp() and calculate_ml(), as well as init_calc to init each calc.

    Parameters
    ----------
    model_name: str
        can be one of: ["gemnet", "spinconv", "dimenetpp"]

    model_path: str
        path to gemnet model config, e.g. '/home/jovyan/working/ocp/configs/s2ef/all/gemnet/gemnet-dT.yml'
        path to spinconv model config, e.g. '/home/jovyan/working/ocp/configs/s2ef/all/spinconv/spinconv_force.yml'
        path to dimenetpp model config, e.g. '/home/jovyan/working/ocp/configs/s2ef/all/dimenet_plus_plus/dpp_forceonly.yml'

    checkpoint_path: str
        path to gemnet model checkpoint, e.g. '/home/jovyan/shared-datasets/OC20/checkpoints/s2ef/gemnet_t_direct_h512_all.pt'
        path to spinconv model checkpoint, e.g. '/home/jovyan/shared-datasets/OC20/checkpoints/s2ef/spinconv_force_centric_all.pt'
        path to dimenetpp model checkpoint, e.g. '/home/jovyan/shared-datasets/OC20/checkpoints/s2ef/dimenetpp_all_forceonly.pt'

    mlp_params: dict
        dictionary of parameters to be passed to be used for initialization of the model/calculator
        should include a 'tuner' key containing a dict with the config specific to this class
        all other keys simply overwrite dicts in the give model_path yml file
    """

    implemented_properties = ["energy", "forces", "stds"]

    def __init__(
        self,
        model_name: str,
        model_path: str,
        checkpoint_path: str,
        mlp_params: dict = {},
    ):

        if model_name not in ["gemnet", "spinconv", "dimenetpp"]:
            raise ValueError("Invalid model name provided")

        if "optimizer" in mlp_params.get("optim", {}):
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            for key in ["optimizer", "scheduler", "ema", "amp"]:
                if key in checkpoint and checkpoint[key] is not None:
                    raise ValueError(
                        str(checkpoint_path)
                        + "\n^this checkpoint contains "
                        + str(key)
                        + " information, please load the .pt file, delete the "
                        + str(key)
                        + " dictionary, save it again as a .pt file, and try again so that the the given optimizer config will be loaded"
                    )

        self.model_name = model_name
        self.model_path = model_path
        self.checkpoint_path = checkpoint_path

        if "tuner" not in mlp_params:
            mlp_params["tuner"] = {}

        config = yaml.safe_load(open(self.model_path, "r"))
        if "includes" in config:
            for include in config["includes"]:
                # Change the path based on absolute path of config_yml
                path = os.path.join(self.model_path.split("configs")[0], include)
                include_config = yaml.safe_load(open(path, "r"))
                config.update(include_config)
        if "optimizer" in mlp_params.get("optim", {}):
            config.pop("optim", None)
        config = merge_dict(config, mlp_params)

        MLPCalc.__init__(self, mlp_params=config)

        self.train_counter = 0
        self.ml_model = False
        self.max_neighbors = self.mlp_params["tuner"].get("max_neighbors", 50)
        self.cutoff = self.mlp_params["tuner"].get("cutoff", 6)
        self.energy_training = self.mlp_params["tuner"].get("energy_training", False)
        if not self.energy_training:
            self.mlp_params["optim"]["energy_coefficient"] = 0
        if "num_threads" in self.mlp_params["tuner"]:
            torch.set_num_threads(self.mlp_params["tuner"]["num_threads"])
        self.validation_split = self.mlp_params["tuner"].get("validation_split", None)

        # init block/weight freezing
        if self.model_name == "gemnet":
            self.unfreeze_blocks = ["out_blocks.3"]
        elif self.model_name == "spinconv":
            self.unfreeze_blocks = ["force_output_block"]
        elif self.model_name == "dimenetpp":
            self.unfreeze_blocks = ["output_blocks.3"]
        if "unfreeze_blocks" in self.mlp_params["tuner"]:
            if isinstance(self.mlp_params["tuner"]["unfreeze_blocks"], list):
                self.unfreeze_blocks = self.mlp_params["tuner"]["unfreeze_blocks"]
            elif isinstance(self.mlp_params["tuner"]["unfreeze_blocks"], str):
                self.unfreeze_blocks = [self.mlp_params["tuner"]["unfreeze_blocks"]]
            else:
                raise ValueError("invalid unfreeze_blocks parameter given")

        # init trainer
        config_dict = copy.deepcopy(self.mlp_params)

        sys.stdout = open(os.devnull, "w")
        self.trainer = Trainer(
            config_yml=config_dict,
            checkpoint=self.checkpoint_path,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
        )
        sys.stdout = sys.__stdout__

    def init_model(self):
        """
        Initialize a new self.trainer containing an ocp ml model using the stored parameter dictionary
        """
        sys.stdout = open(os.devnull, "w")
        self.trainer.load_checkpoint(self.checkpoint_path)
        sys.stdout = sys.__stdout__

        # first freeze all weights within the loaded model
        for name, param in self.trainer.model.named_parameters():
            if param.requires_grad:
                param.requires_grad = False
        # then unfreeze certain weights within the loaded model
        for name, param in self.trainer.model.named_parameters():
            for block_name in self.unfreeze_blocks:
                if block_name in name:
                    param.requires_grad = True

        self.ml_model = True
        self.trainer.train_dataset = GenericDB()

        self.trainer.step = 0
        self.trainer.epoch = 0

    def calculate_ml(self, atoms, properties, system_changes) -> tuple:
        """
        Give ml model the ocp_descriptor to calculate properties : energy, forces, uncertainties.
        overwritable if doing ensembling of ocp calcs

        Args:
            ocp_descriptor: list object containing the descriptor of the atoms object

        Returns:
            tuple: (energy, forces, energy_uncertainty, force_uncertainties)
        """
        e_mean, f_mean = self.trainer.get_atoms_prediction(atoms)

        self.train_counter += 1
        e_std = f_std = self.train_counter * 0.01

        return e_mean, f_mean, e_std, f_std

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Calculate properties including: energy, forces, uncertainties.

        Args:
            atoms: ase Atoms object
        """
        MLPCalc.calculate(
            self, atoms=atoms, properties=properties, system_changes=system_changes
        )

        energy, forces, energy_uncertainty, force_uncertainties = self.calculate_ml(
            atoms, properties, system_changes
        )

        self.results["energy"] = energy
        self.results["forces"] = forces
        self.results["stds"] = [energy_uncertainty, force_uncertainties]
        self.results["force_stds"] = force_uncertainties
        self.results["energy_stds"] = energy_uncertainty
        atoms.info["energy_stds"] = self.results["energy_stds"]

        if atoms.constraints:
            constraints_index = atoms.constraints[0].index
        else:
            constraints_index = []

        abs_force_uncertainty = np.average(
            np.abs(
                np.delete(
                    force_uncertainties,
                    constraints_index,
                    axis=0,
                )
            )
        ).item()

        avg_forces = np.average(
            np.abs(
                np.delete(
                    forces,
                    constraints_index,
                    axis=0,
                )
            )
        ).item()

        atoms.info["max_force_stds"] = abs_force_uncertainty / avg_forces
        # atoms.info["max_force_stds"] = np.nanmax(self.results["force_stds"])
        return

    def train(self, parent_dataset: "list[Atoms]", new_dataset: "list[Atoms]" = None):
        """
        Train the ml model by fitting a new model on the parent dataset,
        or partial fit the current model on just the new_dataset

        Args:
            parent_dataset: list of all the descriptors to be trained on

            new_dataset: list of just the new descriptors to partially fit on
        """
        self.train_counter = 0
        if not self.ml_model or not new_dataset:
            self.init_model()
            dataset = parent_dataset
        else:
            dataset = new_dataset

        start = time.time()
        self.train_ocp(dataset)
        end = time.time()
        print(
            "Time to train "
            + str(self.model_name)
            + " on "
            + str(len(dataset))
            + " pts: "
            + str(end - start)
            + " seconds"
        )

    def train_ocp(self, dataset):
        """
        Overwritable if doing ensembling of ocp models
        """
        if self.validation_split is not None:
            val_indices = []
            for i in self.validation_split:
                val_indices.append(i % len(dataset))
            temp_dataset = []
            valset = []
            for i in range(len(dataset)):
                if i in val_indices:
                    valset.append(dataset[i])
                else:
                    temp_dataset.append(dataset[i])
            dataset = temp_dataset
            val_loader = self.get_data_from_atoms(valset)
            self.trainer.val_loader = val_loader

        train_loader = self.get_data_from_atoms(dataset)
        self.trainer.train_loader = train_loader
        self.trainer.train(disable_eval_tqdm=True)

    def get_data_from_atoms(self, dataset):
        """
        get train_loader object to replace for the ocp model trainer to train on
        """
        a2g = AtomsToGraphs(
            max_neigh=self.max_neighbors,
            radius=self.cutoff,
            r_energy=True,
            r_forces=True,
            r_distances=True,
            r_edges=True,
        )

        graphs_list = [a2g.convert(atoms) for atoms in dataset]

        for graph in graphs_list:
            graph.fid = 0
            graph.sid = 0

        graphs_list_dataset = GraphsListDataset(graphs_list)

        train_sampler = self.trainer.get_sampler(
            graphs_list_dataset, len(dataset), shuffle=False
        )
        self.trainer.train_sampler = train_sampler

        data_loader = self.trainer.get_dataloader(
            graphs_list_dataset,
            train_sampler,
        )

        return data_loader


class GraphsListDataset(Dataset):
    def __init__(self, graphs_list):
        self.graphs_list = graphs_list

    def __len__(self):
        return len(self.graphs_list)

    def __getitem__(self, idx):
        graph = self.graphs_list[idx]
        return graph


class GenericDB:
    def __init__(self):
        pass

    def close_db(self):
        pass


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
            # Only keeps the train data that might have normalizer values
            config["dataset"] = config["dataset"][0]
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
        config["model"]["otf_graph"] = True

        # Save config so obj can be transported over network (pkl)
        self.config = copy.deepcopy(config)
        self.config["checkpoint"] = checkpoint

        if "normalizer" not in config:
            del config["dataset"]["src"]
            config["normalizer"] = config["dataset"]

        super().__init__(
            task=config["task"],
            model=config["model"],
            dataset=None,
            optimizer=config["optim"],
            identifier="",
            normalizer=config["normalizer"],
            slurm=config.get("slurm", {}),
            local_rank=config.get("local_rank", 0),
            is_debug=config.get("is_debug", True),
            cpu=True,
        )

        if checkpoint is not None:
            try:
                self.load_checkpoint(checkpoint)
            except NotImplementedError:
                logging.warning("Unable to load checkpoint!")

        self.a2g = AtomsToGraphs(
            max_neigh=max_neighbors,
            radius=cutoff,
            r_energy=False,
            r_forces=False,
            r_distances=False,
        )

    def get_atoms_prediction(self, atoms):
        data_object = self.a2g.convert(atoms)
        batch = data_list_collater([data_object])
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

                if self.scheduler.scheduler_type == "ReduceLROnPlateau":
                    if self.step % eval_every == 0:
                        self.scheduler.step(
                            metrics=val_metrics[primary_metric]["metric"],
                        )
                else:
                    self.scheduler.step()

            torch.cuda.empty_cache()

            if checkpoint_every == -1:
                self.save(checkpoint_file="checkpoint.pt", training_state=True)

        self.train_dataset.close_db()
        if "val_dataset" in self.config:
            self.val_dataset.close_db()
        if "test_dataset" in self.config:
            self.test_dataset.close_db()
