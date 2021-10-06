#
from ase.atoms import Atoms
from ase.io import Trajectory
from numpy import ndarray
from al_mlp.mongo import MongoWrapper
import ase.db
from ase.calculators.calculator import Calculator
import random
import wandb


class Logger:
    """
    Used by online and offline learners to log their results to wandb, mongodb, and asedb consistently
    """

    def __init__(
        self,
        learner_params: dict,
        ml_potential: Calculator,
        parent_calc: Calculator,
        base_calc: Calculator = None,
        mongo_db_collection=None,
        optional_config: dict = None,
    ):
        """
        Arguments
        ----------
        learner_params: dict
            Dictionary of learner parameters and settings.

        optional_configs: dict
            Optional dictionary of all other configs and settings associated with a run.

        mongo_db: MongoClient
            Optional MongoClient from the pymongo package associated with the desired mongo_db.
        """

        self.step = 0

        # initialize local ASE db file
        self.asedb_name = learner_params.get("asedb_name", "oal_queried_images.db")
        if self.asedb_name is not None:
            ase.db.connect(self.asedb_name, append=False)

        # initialize mongo db
        self.mongo_wrapper = None
        if mongo_db_collection is not None:
            self.mongo_wrapper = MongoWrapper(
                mongo_db_collection,
                learner_params,
                ml_potential,
                parent_calc,
                base_calc,
            )

        # initialize Weights and Biases run
        self.wandb_run = None
        wandb_init = learner_params.get("wandb_init", {})
        if wandb_init.get("wandb_log", False) is True:
            wandb_config = {
                "learner": learner_params,
                "ml_potential": ml_potential.mlp_params,
            }
            if self.mongo_wrapper is not None:
                wandb_config["mongo"] = self.mongo_wrapper.params
            if optional_config is not None:
                wandb_config["run_config"] = optional_config
            self.wandb_run = wandb.init(
                project=wandb_init.get("project", "almlp"),
                name=wandb_init.get("name", "DefaultName"),
                entity=wandb_init.get("entity", "ulissi-group"),
                group=wandb_init.get("group", "DefaultGroup"),
                notes=wandb_init.get("notes", ""),
                config=wandb_config,
            )

        self.pca_metrics = False
        self.uncertainty_metrics = False
        self.parent_traj = None
        # if a trajectory is supplied in the optional config, store that for PCA, uncertainty metrics, etc.
        if optional_config is not None and "links" in optional_config and "traj" in optional_config["links"]:
            self.parent_traj = Trajectory(optional_config["links"]["traj"])
            self.pca_metrics = learner_params.get("logger", {}).get("pca_metrics", False)
            self.uncertainty_metrics = learner_params.get("logger", {}).get("uncertainty_metrics", False)

    def write(self, atoms: Atoms, info: dict):
        if self.pca_metrics:
            pass  # call function to get pca x and y values and store them in info
        if self.uncertainty_metrics:
            pass  # call function to get model uncertainty error and calibration on whole trajectory

        # write to ASE db
        if self.asedb_name is not None:
            random.seed(self.step)
            dict_to_write = {}
            for key, value in info.items():
                if key in ["energy", "fmax", "forces"]:
                    write_key = "reported_" + key
                else:
                    write_key = key

                dict_to_write[write_key] = value
                if value is None:
                    dict_to_write[write_key] = "-"
                elif type(value) is ndarray:
                    dict_to_write[write_key] = str(value)
            with ase.db.connect(self.asedb_name) as asedb:
                asedb.write(
                    atoms,
                    key_value_pairs=dict_to_write,
                    # id=self.step,
                )

        # write to mongo db
        if self.mongo_wrapper is not None:
            self.mongo_wrapper.write_to_mongo(atoms, info)

        # write to Weights and Biases
        if self.wandb_run is not None:
            wandb.log({key: value for key, value in info.items() if value is not None})

        # increment step
        self.step += 1
