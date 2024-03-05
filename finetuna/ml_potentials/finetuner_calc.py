from ase.calculators.calculator import all_changes
from ase.atoms import Atoms
from finetuna.ml_potentials.ml_potential_calc import MLPCalc
import sys, os
from finetuna.job_creator import merge_dict
import copy
import time
import torch
import numpy as np
from finetuna.ocp_models.adapter_gemnet_t import adapter_gemnet_t
from finetuna.finetuner_utils.utils import GenericDB, GraphsListDataset
from finetuna.finetuner_utils.trainer import Trainer
import ocpmodels


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
        checkpoint_path: str,
        mlp_params: dict = {},
    ):
        self.checkpoint_path = checkpoint_path
        mlp_params["checkpoint"] = checkpoint_path
        config = torch.load(self.checkpoint_path, map_location="cpu")["config"]
        self.model_name = config["model"]
        config["model_attributes"]["name"] = config.pop("model")
        config["model"] = config.pop("model_attributes")
        config["trainer"] = "forces"

        if isinstance(config["model"].get("scale_file", None), str):
            scale_file_path = config["model"]["scale_file"]
            if not scale_file_path[0] == "/":
                config["model"]["scale_file"] = (
                    ocpmodels.__file__[:-21] + scale_file_path
                )

        if "tuner" not in mlp_params:
            mlp_params["tuner"] = {}

        if "optimizer" in mlp_params.get("optim", {}):
            config.pop("optim", None)
        config = merge_dict(config, mlp_params)

        MLPCalc.__init__(self, mlp_params=config)

        self.train_counter = 0
        self.max_neighbors = self.mlp_params["tuner"].get("max_neighbors", 50)
        self.cutoff = self.mlp_params["tuner"].get("cutoff", 6)
        self.energy_training = self.mlp_params["tuner"].get("energy_training", False)
        if not self.energy_training:
            self.mlp_params["optim"]["energy_coefficient"] = 0
        if "num_threads" in self.mlp_params["tuner"]:
            torch.set_num_threads(self.mlp_params["tuner"]["num_threads"])
        self.validation_split = self.mlp_params["tuner"].get("validation_split", None)

        self.ref_atoms = None
        self.ref_energy_parent = None
        self.ref_energy_ml = None

        # init block/weight freezing
        self.unfreeze_blocks = self.mlp_params["tuner"].get("unfreeze_blocks", [])
        if isinstance(self.unfreeze_blocks, list):
            pass
        elif isinstance(self.unfreeze_blocks, str):
            self.unfreeze_blocks = [self.unfreeze_blocks]
        else:
            raise ValueError("invalid unfreeze_blocks parameter given")

        # load the self.trainer
        self.load_trainer()

    def load_trainer(self):
        """
        Initialize a new ocpmodels self.trainer (only call once!)
        Can be overwritten by classes that want to use an already instantiated ocpmodels trainer
        """
        # make a copy of the config dict so the trainer doesn't edit the original
        config_dict = copy.deepcopy(self.mlp_params)
        print(config_dict["dataset"])
        print("---------------------------")
        # initialize trainer
        sys.stdout = open(os.devnull, "w")
        self.trainer = Trainer(
            config_yml=config_dict,
            checkpoint_path=self.checkpoint_path,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
        )
        sys.stdout = sys.__stdout__

        # load model for the first time
        self.init_model()

    def init_model(self):
        """
        Initialize a new model in self.trainer using the stored parameter dictionary
        """
        sys.stdout = open(os.devnull, "w")
        self.trainer.load_model()
        self.trainer.load_loss()
        self.trainer.load_optimizer()
        self.trainer.load_extras()
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
        e_std = self.train_counter * 0.01
        f_std = np.zeros_like(f_mean) + (self.train_counter * 0.01)

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

        if self.ref_energy_parent is not None:
            energy += self.ref_energy_parent - self.ref_energy_ml

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
        self.reset()
        if not new_dataset:
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

        if self.ref_energy_parent is not None:
            self.ref_energy_ml, f = self.trainer.get_atoms_prediction(self.ref_atoms)

    def train_ocp(self, dataset):
        """
        Overwritable if doing ensembling of ocp models
        """
        # set the new max epoch to whatever the starting epoch will be + the current max epoch size
        start_epoch = self.trainer.step // len(dataset)
        max_epochs = start_epoch + self.mlp_params["optim"]["max_epochs"]
        self.trainer.config["optim"]["max_epochs"] = int(max_epochs)

        self.trainer.load_optimizer()
        self.trainer.load_extras()

        if (self.validation_split is not None) and (
            len(dataset) > len(self.validation_split)
        ):
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

        graphs_list = [self.trainer.a2g_convert(atoms, True) for atoms in dataset]

        for graph in graphs_list:
            graph.fid = 0
            graph.sid = 0

        graphs_list_dataset = GraphsListDataset(graphs_list)

        train_sampler = self.trainer.get_sampler(
            graphs_list_dataset,
            self.mlp_params.get("optim", {}).get("batch_size", 1),
            shuffle=False,
        )
        self.trainer.train_sampler = train_sampler

        data_loader = self.trainer.get_dataloader(
            graphs_list_dataset,
            train_sampler,
        )

        return data_loader

    def set_lr(self, lr):
        self.trainer.config["optim"]["lr_initial"] = lr

    def set_max_epochs(self, max_epochs):
        self.mlp_params["optim"]["max_epochs"] = max_epochs

    def set_validation(self, val_set: "list[Atoms]"):
        self.trainer.val_loader = self.get_data_from_atoms(val_set)

    def set_test(self, test_set: "list[Atoms]"):
        self.trainer.test_loader = self.get_data_from_atoms(test_set)

    def set_reference_atoms(self, atoms):
        """
        Helper and external function for setting a parent reference energy to correct the ML predicted energy.
        Takes an atoms object with parent singlepoint calculator attached.
        Sets the atoms object as the reference atoms object, and the parent energy as the correspond parent reference energy.
        """
        self.ref_atoms = atoms
        self.ref_energy_parent = self.ref_atoms.get_potential_energy()
        self.ref_energy_ml, f = self.trainer.get_atoms_prediction(self.ref_atoms)
