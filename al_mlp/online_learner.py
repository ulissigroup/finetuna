import copy
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator as sp
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
        n_ensembles=10,
        n_cores="max",
    ):
        Calculator.__init__(self)

        self.n_ensembles = n_ensembles
        self.parent_calc = parent_calc
        self.trainer = trainer
        self.learner_params = learner_params
        self.n_cores = n_cores
        self.parent_dataset = convert_to_singlepoint(parent_dataset)

        # Don't bother making an ensemble with only one data point,
        # as the uncertainty is meaningless
        if len(self.parent_dataset) > 1:
            self.ensemble_sets, self.parent_dataset = non_bootstrap_ensemble(
                parent_dataset, n_ensembles=n_ensembles
            )
            self.ensemble_calc = EnsembleCalc.make_ensemble(
                self.ensemble_sets, self.trainer
            )

        self.uncertain_tol = learner_params["uncertain_tol"]
        self.parent_calls = 0

    def unsafe_prediction(self, atoms, energy_pred, force_pred):

        # Set the desired tolerance based on the current max predcited force
        uncertainty = atoms.info["uncertainty"][0]
        base_uncertainty = np.nanmax(np.abs(force_pred)) ** 2
        uncertainty_tol = self.uncertain_tol * base_uncertainty

        print("uncertainty: %f, uncertainty_tol: %f" % (uncertainty, uncertainty_tol))

        if uncertainty > uncertainty_tol:
            return True
        else:
            return False

    def add_data_and_retrain(self, atoms):
        print("OnlineLearner: Parent calculation required")
        new_data = atoms.copy()
        new_data.set_calculator(copy.copy(self.parent_calc))

        energy_actual = new_data.get_potential_energy(apply_constraint=False)
        force_actual = new_data.get_forces(apply_constraint=False)

        new_data.set_calculator(
            sp(atoms=new_data, energy=energy_actual, forces=force_actual)
        )

        self.ensemble_sets, self.parent_dataset = non_bootstrap_ensemble(
            self.parent_dataset,
            [new_data],
            n_ensembles=self.n_ensembles,
        )

        # Don't bother training if
        if len(self.parent_dataset) > 1:
            self.ensemble_calc = EnsembleCalc.make_ensemble(
                self.ensemble_sets, self.trainer
            )

        self.parent_calls += 1

        return energy_actual, force_actual

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        # If we have less than two data points, uncertainty is not
        # well calibrated so just use DFT
        if len(self.parent_dataset) < 2:
            energy, force = self.add_data_and_retrain(atoms)
            self.results["energy"] = energy
            self.results["forces"] = force
            return

        # Get the predicted energy/force from the ensemble
        energy_pred = self.ensemble_calc.get_potential_energy(atoms)
        force_pred = self.ensemble_calc.get_forces(atoms)

        # Check if we are extrapolating too far, and if so add/retrain
        if self.unsafe_prediction(atoms, energy_pred, force_pred):
            # We ran DFT, so just use that energy/force
            energy, force = self.add_data_and_retrain(atoms)
        else:
            energy, force = energy_pred, force_pred

        # Return the energy/force
        self.results["energy"] = energy
        self.results["forces"] = force
