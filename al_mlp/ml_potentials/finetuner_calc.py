from ase.calculators.calculator import all_changes
from ase.atoms import Atoms
from al_mlp.ml_potentials.ml_potential_calc import MLPCalc
from torch.utils.data import Dataset
from ocpmodels.preprocessing import AtomsToGraphs
import sys, os
from ocpmodels.common.relaxation.ase_utils import OCPCalculator
import yaml
from al_mlp.job_creator import merge_dict
import copy


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
        config = merge_dict(config, mlp_params)

        MLPCalc.__init__(self, mlp_params=config)

        self.ml_model = False
        self.max_neighbors = self.mlp_params["tuner"].get("max_neighbors", 50)
        self.cutoff = self.mlp_params["tuner"].get("cutoff", 6)
        self.energy_training = self.mlp_params["tuner"].get("energy_training", False)
        if not self.energy_training:
            self.mlp_params["optim"]["energy_coefficient"] = 0

    def init_model(self, batch_size):
        """
        Initialize a new self.ocp_calc ml model using the stored parameter dictionary
        """
        # choose blocks to unfreeze based on model
        if self.model_name == "gemnet":
            unfreeze_blocks = "out_blocks.3"
        elif self.model_name == "spinconv":
            unfreeze_blocks = "force_output_block"
        elif self.model_name == "dimenetpp":
            unfreeze_blocks = "output_blocks.3"
        if "unfreeze_blocks" in self.mlp_params["tuner"]:
            unfreeze_blocks = self.mlp_params["tuner"]["unfreeze_blocks"]

        config_dict = copy.deepcopy(self.mlp_params)
        config_dict["optim"]["batch_size"] = batch_size
        config_dict["optim"]["eval_batch_size"] = batch_size

        sys.stdout = open(os.devnull, "w")
        self.ocp_calc = OCPCalculator(
            config_yml=config_dict,
            checkpoint=self.checkpoint_path,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
        )
        sys.stdout = sys.__stdout__

        # freeze certain weights within the loaded model
        for name, param in self.ocp_calc.trainer.model.named_parameters():
            if param.requires_grad:
                if unfreeze_blocks not in name:
                    param.requires_grad = False

        self.ml_model = True
        self.ocp_calc.trainer.train_dataset = GenericDB()

        self.ocp_calc.trainer.step = 0
        self.ocp_calc.trainer.epoch = 0

    def calculate_ml(self, atoms, properties, system_changes) -> tuple:
        """
        Give ml model the ocp_descriptor to calculate properties : energy, forces, uncertainties.
        overwritable if doing ensembling of ocp calcs

        Args:
            ocp_descriptor: list object containing the descriptor of the atoms object

        Returns:
            tuple: (energy, forces, energy_uncertainty, force_uncertainties)
        """
        self.ocp_calc.calculate(atoms, properties, system_changes)
        e_mean = self.ocp_calc.results["energy"]
        f_mean = self.ocp_calc.results["forces"]

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
        atoms.info["max_force_stds"] = self.results["force_stds"]
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
            self.init_model(len(parent_dataset))
            self.train_ocp(parent_dataset)
        else:
            self.train_ocp(new_dataset)

    def train_ocp(self, dataset):
        """
        Overwritable if doing ensembling of ocp calcs
        """
        train_loader = self.get_data_from_atoms(dataset)
        self.ocp_calc.trainer.train_loader = train_loader
        self.ocp_calc.trainer.train()

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

        train_sampler = self.ocp_calc.trainer.get_sampler(
            graphs_list_dataset, 1, shuffle=False
        )
        self.ocp_calc.trainer.train_sampler = train_sampler

        data_loader = self.ocp_calc.trainer.get_dataloader(
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
