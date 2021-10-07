from logging import warn
import numpy as np
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from al_mlp.utils import convert_to_singlepoint, write_to_db_online
import time
import math
import ase.db
from al_mlp.mongo import MongoWrapper
import wandb
import queue

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
        mongo_db=None,
        optional_config=None,
    ):
        Calculator.__init__(self)
        self.parent_calc = parent_calc
        self.ml_potential = ml_potential
        self.learner_params = learner_params
        self.init_learner_params()
        self.parent_dataset = []
        self.complete_dataset = []
        self.queried_db = ase.db.connect("oal_queried_images.db", append=False)
        self.check_final_point = False
        if self.wandb_log is True:
            wandb_config = {
                "learner": self.learner_params,
                "ml_potential": self.ml_potential.mlp_params,
            }
            if mongo_db is not None:
                wandb_config["mongo"] = self.mongo_wrapper.params
            if optional_config is not None:
                wandb_config["run_config"] = optional_config
            self.wandb_run = wandb.init(
                project=self.wandb_init.get("project", "almlp"),
                name=self.wandb_init.get("name", "DefaultName"),
                entity=self.wandb_init.get("entity", "ulissi-group"),
                group=self.wandb_init.get("group", "DefaultGroup"),
                notes=self.wandb_init.get("notes", ""),
                config=wandb_config,
            )
        self.init_mongo(mongo_db=mongo_db)

        self.parent_calls = 0
        self.curr_step = 0

        for image in parent_dataset:
            self.add_data_and_retrain(image)

    def init_mongo(self, mongo_db):
        if mongo_db is not None:
            self.mongo_wrapper = MongoWrapper(
                mongo_db["online_learner"],
                self.learner_params,
                self.ml_potential,
                self.parent_calc,
                None,
            )
        else:
            self.mongo_wrapper = None

    def init_learner_params(self):
        self.fmax_verify_threshold = self.learner_params.get(
            "fmax_verify_threshold", np.nan
        )
        self.stat_uncertain_tol = self.learner_params["stat_uncertain_tol"]
        self.dyn_uncertain_tol = self.learner_params["dyn_uncertain_tol"]
        self.suppress_warnings = self.learner_params.get("suppress_warnings", False)
        self.reverify_with_parent = self.learner_params.get(
            "reverify_with_parent", True
        )
        self.partial_fit = self.learner_params.get("partial_fit", False)
        self.train_on_recent_points = self.learner_params.get(
            "train_on_recent_points", None
        )
        self.num_initial_points = self.learner_params.get("num_initial_points", 2)
        self.initial_points_to_keep = self.learner_params.get(
            "initial_points_to_keep", [i for i in range(self.num_initial_points)]
        )
        self.uncertainty_metric = self.learner_params.get(
            "uncertainty_metric", "forces"
        )
        self.tolerance_selection = self.learner_params.get("tolerance_selection", "max")
        self.no_position_change_steps = self.learner_params.get(
            "no_position_change_steps", None
        )
        if self.no_position_change_steps is not None:
            self.min_position_change = self.learner_params.get(
                "min_position_change", 0.04
            )
            self.positions_queue = queue.Queue(maxsize=self.no_position_change_steps)

        self.rolling_opt_window = self.learner_params.get("rolling_opt_window", None)

        self.constraint = self.learner_params.get("train_on_constraint", False)

        self.wandb_init = self.learner_params.get("wandb_init", {})
        self.wandb_log = self.wandb_init.get("wandb_log", False)

    def init_info(self):
        self.info = {
            "check": None,
            "energy": None,
            "forces": None,
            "fmax": None,
            "ml_energy": None,
            "ml_forces": None,
            "ml_fmax": None,
            "parent_energy": None,
            "parent_forces": None,
            "parent_fmax": None,
            "force_uncertainty": None,
            "energy_uncertainty": None,
            "dyn_uncertainty_tol": None,
            "stat_uncertain_tol": None,
            "tolerance": None,
            "parent_calls": None,
            "energy_error": None,
            "forces_error": None,
        }

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.curr_step += 1

        self.init_info()

        energy, forces, fmax = self.get_energy_and_forces(atoms)

        # Print a statement about the uncertainty
        uncertainty_statement = "uncertainty: "
        if self.uncertainty_metric == "forces":
            uncertainty_statement += str(self.info["force_uncertainty"])
        elif self.uncertainty_metric == "energy":
            uncertainty_statement += str(self.info["energy_uncertainty"])
        uncertainty_statement += ", tolerance: " + str(self.info["tolerance"])
        print(uncertainty_statement)

        # Record number of parent calls after this calculation
        self.info["parent_calls"] = self.parent_calls

        # Return the energy/force
        self.results["energy"] = self.info["energy"] = energy
        self.results["forces"] = forces
        self.info["forces"] = str(forces)
        self.info["fmax"] = fmax

        # Write to asedb, mongodb, wandb
        write_to_db_online(self.queried_db, [atoms], self.info, self.curr_step)
        if self.mongo_wrapper is not None:
            self.mongo_wrapper.write_to_mongo(atoms, self.info)

        if self.wandb_log:
            wandb.log(
                {key: value for key, value in self.info.items() if value is not None}
            )

    def get_energy_and_forces(self, atoms):
        atoms_copy = atoms.copy()

        # If we have less than two data points, uncertainty is not
        # well calibrated so just use DFT
        if len(self.parent_dataset) < self.num_initial_points:
            energy, forces, constrained_forces = self.add_data_and_retrain(atoms_copy)
            fmax = np.sqrt((constrained_forces ** 2).sum(axis=1).max())

            self.info["check"] = True
            self.info["parent_energy"] = energy
            self.info["parent_forces"] = str(forces)
            self.info["parent_fmax"] = fmax

            atoms_copy.info["check"] = True

            if len(self.parent_dataset) == self.num_initial_points:
                new_parent_dataset = [
                    self.parent_dataset[i] for i in self.initial_points_to_keep
                ]
                self.parent_dataset = new_parent_dataset
                self.num_initial_points = len(self.parent_dataset)

        else:
            atoms_ML = self.get_ml_prediction(atoms_copy)
            self.complete_dataset.append(atoms_ML)

            # Get ML potential predicted energies and forces
            energy = atoms_ML.get_potential_energy(apply_constraint=self.constraint)
            forces = atoms_ML.get_forces(apply_constraint=self.constraint)
            constrained_forces = atoms_ML.get_forces()
            fmax = np.sqrt((constrained_forces ** 2).sum(axis=1).max())
            self.info["ml_energy"] = energy
            self.info["ml_forces"] = str(forces)
            self.info["ml_fmax"] = fmax

            # Check if we are extrapolating too far
            need_to_retrain = self.unsafe_prediction(atoms_ML) or self.parent_verify(
                atoms_ML
            )

            self.info["force_uncertainty"] = atoms_ML.info["max_force_stds"]
            self.info["energy_uncertainty"] = atoms_ML.info.get("energy_stds", None)
            self.info["dyn_uncertainty_tol"] = atoms_ML.info["dyn_uncertain_tol"]
            self.info["stat_uncertain_tol"] = atoms_ML.info["stat_uncertain_tol"]
            self.info["tolerance"] = atoms_ML.info["uncertain_tol"]

            # If we are extrapolating too far add/retrain
            if need_to_retrain:
                energy_ML = energy
                constrained_forces_ML = constrained_forces
                # Run DFT, so use that energy/force
                energy, forces, constrained_forces = self.add_data_and_retrain(
                    atoms_copy
                )
                fmax = np.sqrt((constrained_forces ** 2).sum(axis=1).max())

                self.info["check"] = True
                self.info["parent_energy"] = energy
                self.info["parent_forces"] = str(forces)
                self.info["parent_fmax"] = fmax
                self.info["energy_error"] = energy - energy_ML
                self.info["forces_error"] = np.sum(
                    constrained_forces - constrained_forces_ML
                )

                atoms_copy.info["check"] = True
            else:
                # Otherwise use the ML predicted energies and forces
                self.info["check"] = False

                atoms_copy.info["check"] = False

        return energy, forces, fmax

    def unsafe_prediction(self, atoms):
        # Set the desired tolerance based on the current max predcited force or energy
        if self.uncertainty_metric == "forces":
            uncertainty = atoms.info["max_force_stds"]
            if math.isnan(uncertainty):
                raise ValueError("NaN uncertainty")
            forces = atoms.get_forces()
            base_tolerance = np.sqrt((forces ** 2).sum(axis=1).max())
        elif self.uncertainty_metric == "energy":
            uncertainty = atoms.info["energy_stds"]
            energy = atoms.get_potential_energy()
            base_tolerance = energy
        else:
            raise ValueError("invalid uncertainty metric")

        if self.tolerance_selection == "min":
            uncertainty_tol = min(
                [self.dyn_uncertain_tol * base_tolerance, self.stat_uncertain_tol]
            )
        else:
            uncertainty_tol = max(
                [self.dyn_uncertain_tol * base_tolerance, self.stat_uncertain_tol]
            )

        atoms.info["dyn_uncertain_tol"] = self.dyn_uncertain_tol * base_tolerance
        atoms.info["stat_uncertain_tol"] = self.stat_uncertain_tol
        atoms.info["uncertain_tol"] = uncertainty_tol

        prediction_unsafe = uncertainty > uncertainty_tol

        # check if positions have changed enough in the past n steps
        if self.no_position_change_steps is not None:
            new_positions = atoms.get_positions()
            if self.positions_queue.full():
                old_positions = self.positions_queue.get()
                if (
                    np.linalg.norm(new_positions - old_positions)
                    < self.min_position_change
                ):
                    print(
                        "Positions haven't changed by more than "
                        + str(self.min_position_change)
                        + " in "
                        + str(self.no_position_change_steps)
                        + " steps, check with parent"
                    )
                    prediction_unsafe = True
            self.positions_queue.put(new_positions)

        return prediction_unsafe

    def parent_verify(self, atoms):
        forces = atoms.get_forces()
        fmax = np.sqrt((forces ** 2).sum(axis=1).max())

        verify = False
        if fmax <= self.fmax_verify_threshold:
            verify = True
            print("Force below threshold: check with parent")
        if self.check_final_point:
            verify = True
            print("checking final point")
        return verify

    def add_data_and_retrain(self, atoms):

        # don't redo singlepoints if not instructed to reverify and atoms have proper vasp singlepoints attached
        if (
            self.reverify_with_parent is False
            and hasattr(atoms, "calc")
            and atoms.calc is not None
            and (
                type(atoms.calc) is SinglePointCalculator
                or atoms.calc.name == self.parent_calc.name
            )
        ):
            if not self.suppress_warnings:
                warn(
                    "Assuming Atoms object Singlepoint is precalculated (to turn this behavior off: set 'reverify_with_parent' to True)"
                )
            new_data = atoms
        # if verifying (or reverifying) do the singlepoints, and record the time parent calls takes
        else:
            print("OnlineLearner: Parent calculation required")
            start = time.time()

            atoms.set_calculator(self.parent_calc)
            (new_data,) = convert_to_singlepoint([atoms])
            end = time.time()
            print(
                "Time to call parent (call #"
                + str(self.parent_calls)
                + "): "
                + str(end - start)
            )

        partial_dataset = self.add_to_dataset(new_data)
        self.complete_dataset.append(partial_dataset[0])

        self.parent_calls += 1

        # retrain the ml potential
        # if training only on recent points, then check if dataset has become long enough to train on subset
        if (self.train_on_recent_points is not None) and (
            len(self.parent_dataset) > self.train_on_recent_points
        ):
            self.ml_potential.train(self.parent_dataset[-self.train_on_recent_points :])
        # otherwise, if partial fitting, partial fit if not training for the first time
        elif (len(self.parent_dataset) >= self.num_initial_points) and (
            self.partial_fit
        ):
            self.ml_potential.train(self.parent_dataset, partial_dataset)
        # otherwise just train as normal
        else:
            self.ml_potential.train(self.parent_dataset)

        energy_actual = new_data.get_potential_energy(apply_constraint=self.constraint)
        force_actual = new_data.get_forces(apply_constraint=self.constraint)
        force_cons = new_data.get_forces()
        return energy_actual, force_actual, force_cons

    def get_ml_calc(self):
        return self.ml_potential

    def get_ml_prediction(self, atoms):
        """
        Helper function which takes an atoms object with no calc attached.
        Returns it with an ML potential predicted singlepoint.
        Designed to be overwritten by subclasses (DeltaLearner) that modify ML predictions.
        """
        atoms_copy = atoms.copy()
        atoms_copy.set_calculator(self.ml_potential)
        (atoms_ML,) = convert_to_singlepoint([atoms_copy])
        return atoms_ML

    def add_to_dataset(self, new_data):
        """
        Helper function which takes an atoms object with parent singlepoint attached.
        And adds the new parent data to the training set.
        Returns the partial dataset just added.
        Designed to be overwritten by subclasses (DeltaLearner) that modify data added to training set.
        """
        partial_dataset = [new_data]
        self.parent_dataset += partial_dataset
        return partial_dataset
