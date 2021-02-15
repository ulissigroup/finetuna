import copy
import numpy as np
from ase.calculators.calculator import Calculator
from al_mlp.utils import convert_to_singlepoint
from al_mlp.bootstrap import non_bootstrap_ensemble
from al_mlp.ensemble_calc import EnsembleCalc

__author__ = "Muhammed Shuaibi"
__email__ = "mshuaibi@andrew.cmu.edu"


class OnlineLearner(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        learner_params,
        trainer,
        parent_dataset,
        parent_calc,
    ):
        Calculator.__init__(self)

        self.parent_calc = parent_calc
        self.trainer = trainer
        self.learner_params = learner_params
        self.parent_dataset = convert_to_singlepoint(parent_dataset)

        # Don't bother making an ensemble with only one data point,
        # as the uncertainty is meaningless
        if len(self.parent_dataset) > 1:
            self.ensemble_sets, self.parent_dataset = non_bootstrap_ensemble(
                parent_dataset, n_ensembles=self.learner_params["n_ensembles"]
            )
            self.ensemble_calc = EnsembleCalc.make_ensemble(
                self.ensemble_sets, self.trainer
            )
            
        if 'fmax_verify_threshold' in self.learner_params:
            self.fmax_verify_threshold = self.learner_params['fmax_verify_threshold']
        else:
            self.fmax_verify_threshold = np.nan # always False

        self.uncertain_tol = learner_params["uncertain_tol"]
        self.parent_calls = 0

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        # If we have less than two data points, uncertainty is not
        # well calibrated so just use DFT
        if len(self.parent_dataset) < 2:
            energy, force = self.add_data_and_retrain(atoms)
            self.results["energy"] = energy
            self.results["forces"] = force
            return

        # Make a copy of the atoms with ensemble energies as a SP
        atoms_copy = atoms.copy()
        atoms_copy.set_calculator(self.ensemble_calc)
        (atoms_ML,) = convert_to_singlepoint([atoms_copy])

        # Check if we are extrapolating too far, and if so add/retrain
        if self.unsafe_prediction(atoms_ML) or self.parent_verify(atoms_ML):
            # We ran DFT, so just use that energy/force
            energy, force = self.add_data_and_retrain(atoms)
        else:
            energy = atoms_ML.get_potential_energy(apply_constraint=False)
            force = atoms_ML.get_forces(apply_constraint=False)

        # Return the energy/force
        self.results["energy"] = energy
        self.results["forces"] = force

        
    def unsafe_prediction(self, atoms):
        # Set the desired tolerance based on the current max predcited force
        uncertainty = atoms.info["uncertainty"][0] ** 0.5
        base_uncertainty = np.nanmax(np.abs(atoms.get_forces()))
        uncertainty_tol = self.uncertain_tol * base_uncertainty

        print(
            "Max Force Std: %1.3f eV/A, Max Force Threshold: %1.3f eV/A"
            % (uncertainty, uncertainty_tol)
        )

        if uncertainty > uncertainty_tol:
            return True
        else:
            return False
        
    def parent_verify(self, atoms):
        forces = atoms.get_forces()
        fmax = np.sqrt((forces ** 2).sum(axis=1).max())
        return fmax < self.fmax_verify_threshold

    def add_data_and_retrain(self, atoms):
        print("OnlineLearner: Parent calculation required")

        atoms_copy = atoms.copy()
        atoms_copy.set_calculator(copy.copy(self.parent_calc))
        (new_data,) = convert_to_singlepoint([atoms_copy])

        energy_actual = new_data.get_potential_energy(apply_constraint=False)
        force_actual = new_data.get_forces(apply_constraint=False)

        self.ensemble_sets, self.parent_dataset = non_bootstrap_ensemble(
            self.parent_dataset,
            new_data,
            n_ensembles=self.learner_params["n_ensembles"],
        )

        # Don't bother training if we have less than two datapoints
        if len(self.parent_dataset) >= 2:
            self.ensemble_calc = EnsembleCalc.make_ensemble(
                self.ensemble_sets, self.trainer
            )

        self.parent_calls += 1

        return energy_actual, force_actual

