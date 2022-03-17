from al_mlp.online_learner.online_learner import OnlineLearner
from al_mlp.utils import convert_to_singlepoint
import numpy as np
from al_mlp.logger import Logger

__author__ = "Joseph Musielewicz"
__email__ = "al.mlp.package@gmail.com"


class WarmStartLearner(OnlineLearner):
    def __init__(
        self,
        learner_params,
        parent_dataset,
        ml_potential,
        parent_calc,
        base_calc,
        mongo_db=None,
        optional_config=None,
    ):
        self.warm_start_calc = base_calc
        self.warming_up = True

        OnlineLearner.__init__(
            self,
            learner_params,
            parent_dataset,
            ml_potential,
            parent_calc,
            mongo_db=mongo_db,
            optional_config=optional_config,
        )

    def init_logger(self, mongo_db, optional_config):
        self.logger = Logger(
            learner_params=self.learner_params,
            ml_potential=self.ml_potential,
            parent_calc=self.parent_calc,
            base_calc=self.warm_start_calc,
            mongo_db_collection=mongo_db["online_learner"],
            optional_config=optional_config,
        )

    def get_energy_and_forces(self, atoms, precalculated=False):
        if self.warming_up:
            # Make a copy of the atoms with ensemble energies as a SP
            atoms_copy = atoms.copy()
            atoms_copy.set_calculator(self.warm_start_calc)
            (atoms_ML,) = convert_to_singlepoint([atoms_copy])

            # Get base calc predicted energies and forces
            energy = atoms_ML.get_potential_energy(apply_constraint=False)
            forces = atoms_ML.get_forces(apply_constraint=False)
            constrained_forces = atoms_ML.get_forces()
            fmax = np.sqrt((constrained_forces**2).sum(axis=1).max())
            self.info["check"] = False
            self.info["ml_energy"] = energy
            self.info["ml_forces"] = str(forces)
            self.info["ml_fmax"] = fmax

            atoms.info["check"] = False

            if fmax < self.fmax_verify_threshold:
                self.warming_up = False
                energy, forces, fmax = super().get_energy_and_forces(
                    atoms, precalculated=precalculated
                )
            return energy, forces, fmax
        else:
            return super().get_energy_and_forces(atoms, precalculated=precalculated)
