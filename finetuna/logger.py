"""Logging active learning results to wandb, mongodb, and ase db."""

from ase.atoms import Atoms
from ase.io import Trajectory
from numpy import ndarray
from finetuna.mongo import MongoWrapper
import ase.db
from ase.calculators.calculator import Calculator
import random
import wandb
from finetuna.utils import compute_with_calc, copy_images
import math
import numpy as np


class Logger:
    """
    Used by online and offline learners to log their results to
    wandb, mongodb, and asedb consistently.
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
            Optional dictionary of all other configs and settings
            associated with a run.

        mongo_db: MongoClient
            Optional MongoClient from the pymongo package associated
            with the desired mongo_db.
        """
        # save input arguments
        self.learner_params = learner_params
        self.ml_potential = ml_potential
        self.parent_calc = parent_calc
        self.base_calc = base_calc
        self.optional_config = optional_config

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

        self.init_extra_info()

    def init_extra_info(self):
        """initialize booleans for extra calculations,
        e.g.: PCA, uncertainty quantification."""
        self.pca_quantify = False
        self.uncertainty_quantify = False
        self.parent_traj = None
        # if a trajectory is supplied in the optional config,
        # store that for PCA, uncertainty metrics, etc.
        if (
            self.optional_config is not None
            and "links" in self.optional_config
            and "traj" in self.optional_config["links"]
        ):
            self.parent_traj = Trajectory(self.optional_config["links"]["traj"])

            self.pca_quantify = self.learner_params.get("logger", {}).get(
                "pca_quantify", False
            )
            self.uncertainty_quantify = self.learner_params.get("logger", {}).get(
                "uncertainty_quantify", False
            )

            if self.pca_quantify:
                from finetuna.pca import TrajPCA

                self.pca_analyzer = TrajPCA(self.parent_traj)

    def write(self, atoms: Atoms, info: dict, extra_info: dict = {}):
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
            wandb.log(
                {
                    key: value
                    for key, value in {**info, **extra_info}.items()
                    if value is not None
                }
            )

        # increment step
        self.step += 1

    def get_pca(self, atoms: Atoms):
        extra_info = {}
        if self.pca_quantify:
            pca_x, pca_y = self.pca_analyzer.analyze_image(atoms)
            extra_info["pca_x"] = pca_x
            extra_info["pca_y"] = pca_y
        return extra_info

    def get_uncertainty(self, ml_potential: Calculator, check: bool):
        extra_info = {}
        if self.uncertainty_quantify and check:
            force_scores, energy_scores = quantify_uncertainty(
                self.parent_traj, ml_potential
            )
            force_scores.pop("adv_group_calibration")
            energy_scores.pop("adv_group_calibration")
            extra_info["force_scores"] = force_scores
            extra_info["energy_scores"] = energy_scores
        return extra_info


def quantify_uncertainty(traj, model_calc):
    from uncertainty_toolbox.metrics import get_all_metrics

    parent_images = copy_images(traj)
    model_images = compute_with_calc(traj, model_calc)

    initial_energy_diff = (
        model_images[0].get_potential_energy() - parent_images[0].get_potential_energy()
    )

    true_forces = []
    predicted_forces = []
    force_uncertainties = []
    true_energies = []
    predicted_energies = []
    energy_uncertainties = []
    for pi, mi in zip(parent_images, model_images):
        true_forces.append(np.sqrt((pi.get_forces() ** 2).sum(axis=1).max()))
        predicted_forces.append(np.sqrt((mi.get_forces() ** 2).sum(axis=1).max()))
        force_uncertainties.append(mi.info["max_force_stds"])

        if math.isnan(force_uncertainties[-1]):
            raise ValueError("NaN uncertainty")

        true_energies.append(pi.get_potential_energy())
        predicted_energies.append(mi.get_potential_energy() - initial_energy_diff)
        energy_uncertainties.append(mi.info["energy_stds"])

    force_scores = get_all_metrics(
        np.array(predicted_forces),
        np.array(force_uncertainties),
        np.array(true_forces),
        verbose=False,
    )
    energy_scores = get_all_metrics(
        np.array(predicted_energies),
        np.array(energy_uncertainties),
        np.array(true_energies),
        verbose=False,
    )
    return force_scores, energy_scores
