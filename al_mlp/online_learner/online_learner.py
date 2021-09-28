from logging import warn
import numpy as np
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from al_mlp.utils import convert_to_singlepoint, subtract_deltas, write_to_db_online
import time
import math
import ase.db
from al_mlp.calcs import DeltaCalc
from al_mlp.mongo import MongoWrapper
import wandb
import queue

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

        self.base_calc = base_calc
        if self.base_calc is not None:
            self.delta_calc = DeltaCalc(
                [self.ml_potential, self.base_calc],
                "add",
                self.parent_calc.refs,
            )

        self.parent_calls = 0
        self.curr_step = 0

        for image in parent_dataset:
            self.add_data_and_retrain(image)

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

        self.rolling_window = self.learner_params.get("rolling_window", None)

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
            # Make a SP with ml energies
            atoms_copy.set_calculator(self.ml_potential)
            (atoms_ML,) = convert_to_singlepoint([atoms_copy])

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

            # Get ML potential predicted energies and forces
            energy = atoms_ML.get_potential_energy(apply_constraint=False)
            forces = atoms_ML.get_forces(apply_constraint=False)
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
                # Run DFT, so use that energy/force
                energy, forces, constrained_forces = self.add_data_and_retrain(
                    atoms_copy
                )
                fmax = np.sqrt((constrained_forces ** 2).sum(axis=1).max())

                self.info["check"] = True
                self.info["parent_energy"] = energy
                self.info["parent_forces"] = str(forces)
                self.info["parent_fmax"] = fmax

                atoms_copy.info["check"] = True
            else:
                # Otherwise use the ML predicted energies and forces
                self.info["check"] = False

                atoms_copy.info["check"] = False

        self.complete_dataset.append(atoms_copy)

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
        if self.reverify_with_parent is False and (
            type(atoms.calc) is SinglePointCalculator
            or atoms.calc.name == self.parent_calc.name
        ):
            if not self.suppress_warnings:
                warn(
                    "Assuming Atoms object Singlepoint is precalculated (to turn this behavior off: set 'reverify_with_parent' to True)"
                )
            # check if parent calc is a delta, if so: turn the vasp singlepoint into a deltacalc singlepoint
            if type(self.parent_calc) is DeltaCalc:
                (new_data,) = subtract_deltas(
                    [atoms], self.parent_calc.calcs[1], self.parent_calc.refs
                )
            # else just use the atoms as normal
            else:
                new_data = atoms
        # if verifying (or reverifying) do the singlepoints, and record the time parent calls takes
        else:
            print("OnlineLearner: Parent calculation required")
            start = time.time()

            atoms_copy = atoms.copy()
            atoms_copy.set_calculator(self.parent_calc)
            (new_data,) = convert_to_singlepoint([atoms_copy])
            end = time.time()
            print(
                "Time to call parent (call #"
                + str(self.parent_calls)
                + "): "
                + str(end - start)
            )

        self.parent_dataset += [new_data]

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
            self.ml_potential.train(self.parent_dataset, [new_data])
        # otherwise just train as normal
        else:
            self.ml_potential.train(self.parent_dataset)

        # if we are using a delta calc, add back on the base calc
        if self.base_calc is not None:
            atoms_copy = atoms.copy()
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
