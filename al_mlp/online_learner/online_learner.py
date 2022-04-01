from logging import warn
import numpy as np
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from al_mlp.logger import Logger
from al_mlp.utils import convert_to_singlepoint, convert_to_top_k_forces
import time
import math
import ase.db
import queue
import os

__author__ = "Joseph Musielewicz"
__email__ = "al.mlp.package@gmail.com"


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
        self.queried_db = ase.db.connect(self.db_name, append=False)
        self.trained_at_least_once = False
        self.check_final_point = False
        self.uncertainty_history = []

        if mongo_db is None:
            mongo_db = {"online_learner": None}
        self.init_logger(mongo_db, optional_config)

        self.parent_calls = 0
        self.curr_step = 0
        self.steps_since_last_query = 0

        for image in parent_dataset:
            self.get_energy_and_forces(image, precalculated=True)

    def init_logger(self, mongo_db, optional_config):
        self.logger = Logger(
            learner_params=self.learner_params,
            ml_potential=self.ml_potential,
            parent_calc=self.parent_calc,
            base_calc=None,
            mongo_db_collection=mongo_db["online_learner"],
            optional_config=optional_config,
        )

    def init_learner_params(self):
        self.fmax_verify_threshold = self.learner_params.get(
            "fmax_verify_threshold", np.nan
        )
        self.stat_uncertain_tol = self.learner_params.get(
            "stat_uncertain_tol", 1000000000
        )
        self.dyn_uncertain_tol = self.learner_params.get(
            "dyn_uncertain_tol", 1000000000
        )
        self.dyn_avg_steps = self.learner_params.get("dyn_avg_steps", None)

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

        self.query_every_n_steps = self.learner_params.get("query_every_n_steps", None)
        self.query_n_fmae_coefficient = self.learner_params.get(
            "query_n_fmae_coefficient", None
        )
        self.train_on_top_k_forces = self.learner_params.get(
            "train_on_top_k_forces", None
        )

        self.store_complete_dataset = self.learner_params.get(
            "store_complete_dataset", False
        )

        self.db_name = self.learner_params.get("asedb_name", "oal_queried_images.db")

        self.wandb_init = self.learner_params.get("wandb_init", {})
        self.wandb_log = self.wandb_init.get("wandb_log", False)

    def set_query_reason(self, reason: str):
        if reason == "final":
            self.info["query"] = -2  # Set to -2 if querying final point
        elif reason == "pretrain":
            self.info["query"] = -1  # Set to -1 if querying before training
        elif reason == "noquery":
            self.info["query"] = 0  # Set to 0 if not querying
        elif reason == "threshold":
            self.info["query"] = 1  # Set to 1 if querying b/c force is below threshold
        elif reason == "static":
            self.info["query"] = 2  # Set to 2 if querying b/c of static tolerance
        elif reason == "dynamic":
            self.info["query"] = 3  # Set to 3 if querying b/c of dynamic tolerance
        elif reason == "position":
            self.info["query"] = 4  # Set to 4 if querying b/c positions not changed
        elif reason == "nsteps":
            self.info["query"] = 5  # Set to 5 if querying b/c it has been n steps
        else:
            raise ValueError("invalid query reason given (" + str(reason) + ")")

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
            "relative_energy_error": None,
            "forces_error": None,
            "relative_forces_error": None,
            "current_step": None,
            "steps_since_last_query": None,
            "query": None,
            "training_time": None,
            "parent_time": None,
            "forces_mae": None,
        }

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.curr_step += 1
        self.steps_since_last_query += 1

        energy, forces, fmax = self.get_energy_and_forces(atoms)
        self.results["energy"] = energy
        self.results["forces"] = forces

        # Print a statement about the uncertainty
        uncertainty_statement = "uncertainty: "
        if self.uncertainty_metric == "forces":
            uncertainty_statement += str(self.info["force_uncertainty"])
        elif self.uncertainty_metric == "energy":
            uncertainty_statement += str(self.info["energy_uncertainty"])
        uncertainty_statement += ", tolerance: " + str(self.info["tolerance"])
        print(uncertainty_statement)

    def get_energy_and_forces(self, atoms, precalculated=False):
        # copy the atoms object (original only used to obtain indices of constraints and for precalculated images)
        atoms_copy = atoms.copy()

        # initialize info dict before doing anything else, such as setting a query reason
        self.init_info()

        # if adding precalculated atoms to parent dataset and trajectory
        if precalculated:
            atoms_copy.calc = atoms.calc
            self.set_query_reason("pretrain")

        # If we have less than two data points, uncertainty is not
        # well calibrated so just use DFT
        if len(self.parent_dataset) < self.num_initial_points:
            atoms_copy.info["check"] = True

            energy, forces, constrained_forces = self.add_data_and_retrain(atoms_copy)
            fmax = np.sqrt((constrained_forces**2).sum(axis=1).max())

            self.info["check"] = True
            self.info["parent_energy"] = energy
            self.info["parent_forces"] = str(forces)
            self.info["parent_fmax"] = fmax
            self.set_query_reason("pretrain")

        else:
            atoms_ML = self.get_ml_prediction(atoms_copy)

            # Get ML potential predicted energies and forces
            energy = atoms_ML.get_potential_energy(apply_constraint=self.constraint)
            forces = atoms_ML.get_forces(apply_constraint=self.constraint)
            constrained_forces = atoms_ML.get_forces()
            fmax = np.sqrt((constrained_forces**2).sum(axis=1).max())
            self.info["ml_energy"] = energy
            self.info["ml_forces"] = str(forces)
            self.info["ml_fmax"] = fmax

            # Check if we are extrapolating too far
            unsafe_bool = self.unsafe_prediction(atoms_ML)
            verify_bool = self.parent_verify(atoms_ML)
            need_to_retrain = unsafe_bool or verify_bool or precalculated

            self.info["force_uncertainty"] = atoms_ML.info["max_force_stds"]
            self.info["energy_uncertainty"] = atoms_ML.info.get("energy_stds", None)
            self.info["dyn_uncertainty_tol"] = atoms_ML.info["dyn_uncertain_tol"]
            self.info["stat_uncertain_tol"] = atoms_ML.info["stat_uncertain_tol"]
            self.info["tolerance"] = atoms_ML.info["uncertain_tol"]

            # If we are extrapolating too far add/retrain
            if need_to_retrain:
                atoms_copy.info["check"] = True

                energy_ML = energy
                constrained_forces_ML = constrained_forces
                # Run DFT, so use that energy/force
                energy, forces, constrained_forces = self.add_data_and_retrain(
                    atoms_copy
                )
                fmax = np.sqrt((constrained_forces**2).sum(axis=1).max())

                self.info["check"] = True
                self.info["parent_energy"] = energy
                self.info["parent_forces"] = str(forces)
                self.info["parent_fmax"] = fmax
                self.info["energy_error"] = energy - energy_ML
                self.info["relative_energy_error"] = (energy - energy_ML) / energy
                self.info["forces_error"] = np.sum(
                    np.abs(constrained_forces - constrained_forces_ML)
                )
                self.info["forces_mae"] = np.mean(
                    np.abs(constrained_forces - constrained_forces_ML)
                )

                if atoms.constraints:
                    constraints_index = atoms.constraints[0].index
                else:
                    constraints_index = []
                self.info["relative_forces_error"] = np.divide(
                    np.sum(
                        np.abs(
                            np.delete(
                                constrained_forces - constrained_forces_ML,
                                constraints_index,
                                axis=0,
                            )
                        )
                    ),
                    np.sum(
                        np.abs(
                            np.delete(
                                constrained_forces,
                                constraints_index,
                                axis=0,
                            )
                        )
                    ),
                ).item()

                # Using retrained ML potential, get new predicted energies and forces
                retrained_atoms_ML = self.get_ml_prediction(atoms_copy)
                retrained_energy = retrained_atoms_ML.get_potential_energy(
                    apply_constraint=self.constraint
                )
                retrained_forces = retrained_atoms_ML.get_forces(
                    apply_constraint=self.constraint
                )
                retrained_constrained_forces = retrained_atoms_ML.get_forces()
                retrained_fmax = np.sqrt(
                    (retrained_constrained_forces**2).sum(axis=1).max()
                )
                self.info["retrained_energy"] = retrained_energy
                self.info["retrained_forces"] = str(retrained_forces)
                self.info["retrained_fmax"] = retrained_fmax
                self.info["retrained_force_error"] = np.sum(
                    np.abs(constrained_forces - retrained_constrained_forces)
                )

            else:
                # Otherwise use the ML predicted energies and forces
                if self.store_complete_dataset:
                    self.complete_dataset.append(atoms_ML)
                else:
                    self.complete_dataset = [atoms_ML]

                self.info["check"] = False
                self.set_query_reason("noquery")

                atoms_copy.info["check"] = False

        # Record number of parent calls after this calculation
        self.info["parent_calls"] = self.parent_calls
        self.info["current_step"] = self.curr_step
        self.info["steps_since_last_query"] = self.steps_since_last_query

        # Return the energy/force
        self.info["energy"] = energy
        self.info["forces"] = str(forces)
        self.info["fmax"] = fmax

        extra_info = {}
        extra_info.update(self.logger.get_pca(atoms))
        if self.trained_at_least_once:
            extra_info.update(
                self.logger.get_uncertainty(self.get_ml_calc(), self.info["check"])
            )
        self.logger.write(atoms, self.info, extra_info=extra_info)

        return energy, forces, fmax

    def unsafe_prediction(self, atoms):
        # Set the desired tolerance based on the current max predicted force or energy
        if self.uncertainty_metric == "forces":
            uncertainty = atoms.info["max_force_stds"]
            if math.isnan(uncertainty):
                raise ValueError("NaN uncertainty")
            forces = atoms.get_forces()
            base_tolerance = np.sqrt((forces**2).sum(axis=1).max())
        elif self.uncertainty_metric == "energy":
            uncertainty = atoms.info["energy_stds"]
            energy = atoms.get_potential_energy()
            base_tolerance = energy
        else:
            raise ValueError("invalid uncertainty metric")

        self.uncertainty_history.append(uncertainty)

        # if we are taking the dynamic uncertainty tolerance to be the average of the past n uncertainties,
        # then calculate that everage and set it as the base tolerance (to be modified by dyn modifier)
        if self.dyn_avg_steps is not None:
            base_tolerance = np.mean(
                self.uncertainty_history[
                    max(0, len(self.uncertainty_history) - self.dyn_avg_steps) :
                ]
            )

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

        if uncertainty > uncertainty_tol:
            if uncertainty > atoms.info["dyn_uncertain_tol"]:
                self.set_query_reason("dynamic")
            if uncertainty > atoms.info["stat_uncertain_tol"]:
                self.set_query_reason("static")

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
                    self.set_query_reason("position")
            self.positions_queue.put(new_positions)

        if self.query_n_fmae_coefficient is not None:
            forces_mae = 0.1
            if hasattr(self, "info") and self.info.get("forces_mae", None) is not None:
                forces_mae = self.info["forces_mae"]
            self.query_every_n_steps = self.query_n_fmae_coefficient * forces_mae

        if self.query_every_n_steps is not None:
            if self.steps_since_last_query > self.query_every_n_steps:
                print(
                    str(self.steps_since_last_query)
                    + " steps since last query, querying every "
                    + str(self.query_every_n_steps)
                    + " so check with parent"
                )
                prediction_unsafe = True
                self.set_query_reason("nsteps")

        return prediction_unsafe

    def parent_verify(self, atoms):
        forces = atoms.get_forces()
        fmax = np.sqrt((forces**2).sum(axis=1).max())

        verify = False
        if (
            fmax <= self.fmax_verify_threshold
            or atoms.info.get("parent_calculation_required", False)
        ):
            verify = True
            print("Force below threshold: check with parent")
            self.set_query_reason("threshold")
        if self.check_final_point:
            verify = True
            print("checking final point")
            self.set_query_reason("final")
        return verify

    def add_data_and_retrain(self, atoms):
        self.steps_since_last_query = 0

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
            self.info["parent_time"] = end - start

        # add to complete dataset (for atomistic methods optimizer replay)
        if self.store_complete_dataset:
            self.complete_dataset.append(new_data)
        else:
            self.complete_dataset = [new_data]

        # before adding to parent (training) dataset, convert to top k forces if applicable
        if self.train_on_top_k_forces is not None:
            [training_data] = convert_to_top_k_forces(
                [new_data], self.train_on_top_k_forces
            )
        else:
            training_data = new_data

        # add to parent dataset (for training) and return partial dataset (for partial fit)
        partial_dataset = self.add_to_dataset(training_data)

        self.parent_calls += 1

        start = time.time()
        # retrain the ml potential only if there is more than enough data that the ml potential may be used
        if len(self.parent_dataset) > self.num_initial_points:
            # if training only on recent points, and have trained before, then check if dataset has become long enough to train on subset
            if (
                self.trained_at_least_once
                and (self.train_on_recent_points is not None)
                and (len(self.parent_dataset) > self.train_on_recent_points)
            ):
                self.ml_potential.train(
                    self.parent_dataset[-self.train_on_recent_points :]
                )
            # otherwise, if partial fitting, partial fit if not training for the first time
            elif (
                self.trained_at_least_once
                and (self.train_on_recent_points is None)
                and (self.partial_fit)
            ):
                self.ml_potential.train(self.parent_dataset, partial_dataset)
            # otherwise just train as normal
            else:
                self.ml_potential.train(self.parent_dataset)
                self.trained_at_least_once = True

        # if the data requirement has just been met: train for the first time on only the initial points to keep
        elif len(self.parent_dataset) == self.num_initial_points:
            new_parent_dataset = [
                self.parent_dataset[i] for i in self.initial_points_to_keep
            ]
            self.parent_dataset = new_parent_dataset
            self.num_initial_points = len(self.parent_dataset)

            self.ml_potential.train(self.parent_dataset)
            self.trained_at_least_once = True
        end = time.time()
        self.info["training_time"] = end - start

        # set the energy and force results of the parent calculator and return them
        energy_actual = new_data.get_potential_energy(apply_constraint=self.constraint)
        force_actual = new_data.get_forces(apply_constraint=self.constraint)
        force_cons = new_data.get_forces()
        return energy_actual, force_actual, force_cons

    def get_ml_calc(self):
        self.ml_potential.reset()
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
