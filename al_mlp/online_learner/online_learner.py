import copy
import numpy as np
from ase.calculators.calculator import Calculator
from al_mlp.utils import convert_to_singlepoint
import time

__author__ = "Muhammed Shuaibi"
__email__ = "mshuaibi@andrew.cmu.edu"


class OnlineLearner(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        learner_params,
        parent_dataset,
        ml_potential,
        parent_calc,
    ):
        Calculator.__init__(self)

        self.parent_calc = parent_calc
        self.learner_params = learner_params
        self.parent_dataset = convert_to_singlepoint(parent_dataset)

        self.ml_potential = ml_potential

        # Don't bother training with only one data point,
        # as the uncertainty is meaningless
        if len(self.parent_dataset) > 1:
            ml_potential.train(self.parent_dataset)

        if "fmax_verify_threshold" in self.learner_params:
            self.fmax_verify_threshold = self.learner_params["fmax_verify_threshold"]
        else:
            self.fmax_verify_threshold = np.nan  # always False

        self.stat_uncertain_tol = learner_params["stat_uncertain_tol"]
        self.dyn_uncertain_tol = learner_params["dyn_uncertain_tol"]
        self.parent_calls = 0
        self.retrain_idx = []
        self.curr_step = 0
        self.unsafe_list = {}
        self.force_list = []

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        # If we have less than two data points, uncertainty is not
        # well calibrated so just use DFT
        if len(self.parent_dataset) < 2:
            energy, force = self.add_data_and_retrain(atoms)
            self.results["energy"] = energy
            self.results["forces"] = force
            self.curr_step += 1
            return

        # Make a copy of the atoms with ensemble energies as a SP
        atoms_ML = atoms.copy()
        atoms_ML.set_calculator(self.ml_potential)
        self.ml_potential.calculate(atoms_ML, properties, system_changes)

        #         (atoms_ML,) = convert_to_singlepoint([atoms_copy])

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
        self.curr_step += 1

    def unsafe_prediction(self, atoms):
        # Set the desired tolerance based on the current max predcited force
        uncertainty = atoms.info["max_force_stds"]
        base_uncertainty = np.nanmax(np.abs(atoms.get_forces()))
        uncertainty_tol = max(
            [self.dyn_uncertain_tol * base_uncertainty, self.stat_uncertain_tol]
        )

        # print(
        #     "Max Force Std: %1.3f eV/A, Max Force Threshold: %1.3f eV/A"
        #     % (uncertainty, uncertainty_tol)
        # )

        # print(
        #     "static tol: %1.3f eV/A, dynamic tol: %1.3f eV/A"
        #     % (self.stat_uncertain_tol, self.dyn_uncertain_tol * base_uncertainty)
        # )
        if uncertainty > uncertainty_tol:
            maxf = np.nanmax(np.abs(atoms.get_forces(apply_constraint=False)))
            self.unsafe_list[self.curr_step] = [maxf, uncertainty, uncertainty_tol]
            return True
        else:
            return False

    def parent_verify(self, atoms):
        forces = atoms.get_forces()
        fmax = np.sqrt((forces ** 2).sum(axis=1).max())
        # fmax = np.max(np.abs(forces))
        # print("fmax ", fmax, "/n")
        # print("verify threshold ", self.fmax_verify_threshold)
        self.force_list.append(self.curr_step)

        if fmax <= self.fmax_verify_threshold:
            print("Force below threshold: check with parent")
        return fmax <= self.fmax_verify_threshold

    def add_data_and_retrain(self, atoms):
        print("OnlineLearner: Parent calculation required")
        start = time.time()
        self.retrain_idx.append(self.curr_step)

        atoms_copy = atoms.copy()
        atoms_copy.set_calculator(self.parent_calc)
        print(atoms_copy)
        (new_data,) = convert_to_singlepoint([atoms_copy])

        energy_actual = new_data.get_potential_energy(apply_constraint=False)
        force_actual = new_data.get_forces(apply_constraint=False)

        self.parent_dataset += [new_data]

        # Don't bother training if we have less than two datapoints
        if len(self.parent_dataset) >= 2:
            self.ml_potential.train(self.parent_dataset, [new_data])
        else:
            self.ml_potential.train(self.parent_dataset)

        self.parent_calls += 1
        end = time.time()
        print("Time to call parent: "+str(end - start))

        return energy_actual, force_actual
