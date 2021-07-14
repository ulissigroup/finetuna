import numpy as np
from ase.calculators.calculator import Calculator
from al_mlp.utils import convert_to_singlepoint, write_to_db_online
import time
import math
import ase.db
import random
from al_mlp.calcs import DeltaCalc
from al_mlp.mongo import MongoWrapper

__author__ = "Muhammed Shuaibi"
__email__ = "mshuaibi@andrew.cmu.edu"


class OnlineLearner(Calculator):
    """
    If base_calc is set to some calculator, OnlineLearner will assume that the ml_potential is some kind of subtracting DeltaCalc. It will add base_calc to all the results.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        learner_params,
        parent_dataset,
        ml_potential,
        parent_calc,
        base_calc=None,
        mongo_db=None,
    ):
        Calculator.__init__(self)
        self.parent_calc = parent_calc
        self.learner_params = learner_params
        self.parent_dataset = convert_to_singlepoint(parent_dataset)
        ase.db.connect("oal_queried_images.db", append=False)
        self.queried_db = ase.db.connect("oal_queried_images.db")
        if mongo_db is not None:
            self.mongo_wrapper = MongoWrapper(
                mongo_db["online_learner"],
                learner_params,
                ml_potential,
                parent_calc,
                base_calc,
            )
        else:
            self.mongo_wrapper = None
        self.ml_potential = ml_potential

        self.base_calc = base_calc
        if self.base_calc is not None:
            self.delta_calc = DeltaCalc(
                [self.ml_potential, self.base_calc],
                "add",
                self.parent_calc.refs,
            )

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
        self.curr_step = 0

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        # If we have less than two data points, uncertainty is not
        # well calibrated so just use DFT
        if len(self.parent_dataset) < 2:
            energy, force, force_cons = self.add_data_and_retrain(atoms)
            parent_fmax = np.sqrt((force_cons ** 2).sum(axis=1).max())
            self.results["energy"] = energy
            self.results["forces"] = force
            self.curr_step += 1
            random.seed(self.curr_step)
            info = {
                "check": True,
                "parentE": energy,
                "parentMaxForce": parent_fmax,
                "parentF": str(force),
            }
            write_to_db_online(
                self.queried_db,
                [atoms],
                info,
            )
            if self.mongo_wrapper is not None:
                self.mongo_wrapper.write_to_mongo(atoms, info)
            return

        # Make a copy of the atoms with ensemble energies as a SP
        atoms_copy = atoms.copy()
        atoms_copy.set_calculator(self.ml_potential)
        (atoms_ML,) = convert_to_singlepoint([atoms_copy])
        self.curr_step += 1

        if self.base_calc is not None:
            new_delta = DeltaCalc(
                [atoms_ML.calc, self.base_calc],
                "add",
                self.parent_calc.refs,
            )
            atoms_copy.set_calculator(new_delta)
            (atoms_delta,) = convert_to_singlepoint([atoms_copy])
            for key, value in atoms_ML.info.items():
                atoms_delta.info[key] = value
            atoms_ML = atoms_delta

        # Check if we are extrapolating too far, and if so add/retrain
        if self.unsafe_prediction(atoms_ML) or self.parent_verify(atoms_ML):
            # We ran DFT, so just use that energy/force
            energy, force, force_cons = self.add_data_and_retrain(atoms)
            parent_fmax = np.sqrt((force_cons ** 2).sum(axis=1).max())
            random.seed(self.curr_step)
            info = {
                "check": True,
                "uncertainty": atoms_ML.info["max_force_stds"],
                "tolerance": atoms_ML.info["uncertain_tol"],
                "dyn_uncertainty_tol": atoms_ML.info["dyn_uncertain_tol"],
                "stat_uncertain_tol": atoms_ML.info["stat_uncertain_tol"],
                "parentE": energy,
                "parentMaxForce": parent_fmax,
                "parentF": str(force),
                "oalF": str(atoms_ML.get_forces()),
            }
            write_to_db_online(
                self.queried_db,
                [atoms_ML],
                info,
            )
            if self.mongo_wrapper is not None:
                self.mongo_wrapper.write_to_mongo(atoms_ML, info)
        else:
            energy = atoms_ML.get_potential_energy(apply_constraint=False)
            force = atoms_ML.get_forces(apply_constraint=False)
            random.seed(self.curr_step)
            info = {
                "check": False,
                "uncertainty": atoms_ML.info["max_force_stds"],
                "dyn_uncertainty_tol": atoms_ML.info["dyn_uncertain_tol"],
                "stat_uncertain_tol": atoms_ML.info["stat_uncertain_tol"],
                "tolerance": atoms_ML.info["uncertain_tol"],
                "oalF": str(force),
            }
            write_to_db_online(
                self.queried_db,
                [atoms_ML],
                info,
            )
            if self.mongo_wrapper is not None:
                self.mongo_wrapper.write_to_mongo(atoms_ML, info)

        # Return the energy/force
        self.results["energy"] = energy
        self.results["forces"] = force

    def unsafe_prediction(self, atoms):
        # Set the desired tolerance based on the current max predcited force
        uncertainty = atoms.info["max_force_stds"]
        if math.isnan(uncertainty):
            raise ValueError("Input is not a positive integer")
        forces = atoms.get_forces(apply_constraint=False)
        base_uncertainty = np.sqrt((forces ** 2).sum(axis=1).max())
        uncertainty_tol = max(
            [self.dyn_uncertain_tol * base_uncertainty, self.stat_uncertain_tol]
        )
        atoms.info["dyn_uncertain_tol"] = self.dyn_uncertain_tol * base_uncertainty
        atoms.info["stat_uncertain_tol"] = self.stat_uncertain_tol
        atoms.info["uncertain_tol"] = uncertainty_tol
        # print(
        #     "Max Force Std: %1.3f eV/A, Max Force Threshold: %1.3f eV/A"
        #     % (uncertainty, uncertainty_tol)
        # )

        # print(
        #     "static tol: %1.3f eV/A, dynamic tol: %1.3f eV/A"
        #     % (self.stat_uncertain_tol, self.dyn_uncertain_tol * base_uncertainty)
        # )
        return uncertainty > uncertainty_tol

    def parent_verify(self, atoms):
        forces = atoms.get_forces()
        fmax = np.sqrt((forces ** 2).sum(axis=1).max())

        if fmax <= self.fmax_verify_threshold:
            print("Force below threshold: check with parent")
        return fmax <= self.fmax_verify_threshold

    def add_data_and_retrain(self, atoms):
        print("OnlineLearner: Parent calculation required")

        start = time.time()

        atoms_copy = atoms.copy()
        atoms_copy.set_calculator(self.parent_calc)
        print(atoms_copy)
        (new_data,) = convert_to_singlepoint([atoms_copy])

        self.parent_dataset += [new_data]

        self.parent_calls += 1

        end = time.time()
        print(
            "Time to call parent (call #"
            + str(self.parent_calls)
            + "): "
            + str(end - start)
        )

        # Don't bother training if we have less than two datapoints
        if len(self.parent_dataset) >= 2:
            self.ml_potential.train(self.parent_dataset, [new_data])
        else:
            self.ml_potential.train(self.parent_dataset)

        if self.base_calc is not None:
            new_delta = DeltaCalc(
                [new_data.calc, self.base_calc],
                "add",
                self.parent_calc.refs,
            )
            atoms_copy.set_calculator(new_delta)
            (new_data,) = convert_to_singlepoint([atoms_copy])

        energy_actual = new_data.get_potential_energy(apply_constraint=False)
        force_actual = new_data.get_forces(apply_constraint=False)
        force_cons = new_data.get_forces()
        return energy_actual, force_actual, force_cons
