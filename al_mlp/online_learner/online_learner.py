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
        self.parent_dataset = []
        self.queried_db = ase.db.connect("oal_queried_images.db", append=False)
        self.check_final_point = False

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

        self.uncertainty_metric = self.learner_params.get(
            "uncertainty_metric", "forces"
        )

        self.wandb_init = self.learner_params.get("wandb_init", {})
        self.wandb_log = self.wandb_init.get("wandb_log", False)
        if self.wandb_log is True:
            wandb_config = {
                "learner": self.learner_params,
                "ml_potential": self.ml_potential.mlp_params,
            }
            if mongo_db is not None:
                wandb_config["mongo"] = self.mongo_wrapper.params
            self.wandb_run = wandb.init(
                project=self.wandb_init.get("project", "almlp"),
                name=self.wandb_init.get("name", "DefaultName"),
                entity=self.wandb_init.get("entity", "ulissi-group"),
                group=self.wandb_init.get("group", "DefaultGroup"),
                notes=self.wandb_init.get("notes", ""),
                config=wandb_config,
            )

        self.base_calc = base_calc
        if self.base_calc is not None:
            self.delta_calc = DeltaCalc(
                [self.ml_potential, self.base_calc],
                "add",
                self.parent_calc.refs,
            )

        if "fmax_verify_threshold" in self.learner_params:
            self.fmax_verify_threshold = self.learner_params["fmax_verify_threshold"]
        else:
            self.fmax_verify_threshold = np.nan  # always False

        self.stat_uncertain_tol = self.learner_params["stat_uncertain_tol"]
        self.dyn_uncertain_tol = self.learner_params["dyn_uncertain_tol"]

        self.parent_calls = 0
        self.curr_step = 0

        for image in parent_dataset:
            self.add_data_and_retrain(image)

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.curr_step += 1

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

        # If we have less than two data points, uncertainty is not
        # well calibrated so just use DFT
        if len(self.parent_dataset) < 2:
            energy, forces, constrained_forces = self.add_data_and_retrain(atoms)
            fmax = np.sqrt((constrained_forces ** 2).sum(axis=1).max())
            self.info["check"] = True
            self.info["ml_energy"] = self.info["parent_energy"] = energy
            self.info["ml_forces"] = self.info["parent_forces"] = str(forces)
            self.info["ml_fmax"] = self.info["parent_fmax"] = fmax
        else:
            # Make a copy of the atoms with ensemble energies as a SP
            atoms_copy = atoms.copy()
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
                energy, forces, constrained_forces = self.add_data_and_retrain(atoms)
                fmax = np.sqrt((constrained_forces ** 2).sum(axis=1).max())
                self.info["check"] = True
                self.info["parent_energy"] = energy
                self.info["parent_forces"] = str(forces)
                self.info["parent_fmax"] = fmax
            else:
                # Otherwise use the ML predicted energies and forces
                self.info["check"] = False

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

        uncertainty_tol = max(
            [self.dyn_uncertain_tol * base_tolerance, self.stat_uncertain_tol]
        )
        atoms.info["dyn_uncertain_tol"] = self.dyn_uncertain_tol * base_tolerance
        atoms.info["stat_uncertain_tol"] = self.stat_uncertain_tol
        atoms.info["uncertain_tol"] = uncertainty_tol
        # print(
        #     "Max Force Std: %1.3f eV/A, Max Force Threshold: %1.3f eV/A"
        #     % (uncertainty, uncertainty_tol)
        # )

        # print(
        #     "static tol: %1.3f eV/A, dynamic tol: %1.3f eV/A"
        #     % (self.stat_uncertain_tol, self.dyn_uncertain_tol * base_tolerance)
        # )
        return uncertainty > uncertainty_tol

    def parent_verify(self, atoms):
        forces = atoms.get_forces()
        fmax = np.sqrt((forces ** 2).sum(axis=1).max())

        if fmax <= self.fmax_verify_threshold:
            print("Force below threshold: check with parent")
        if self.check_final_point:
            print("checking final point")
        verify = (fmax <= self.fmax_verify_threshold) or self.check_final_point
        return verify

    def add_data_and_retrain(self, atoms):
        print("OnlineLearner: Parent calculation required")

        start = time.time()

        # don't redo singlepoints if not instructed to reverify and atoms have proper vasp singlepoints attached
        if (
            self.learner_params.get("reverify_with_parent", True) is False
            and type(atoms.calc) is SinglePointCalculator
            and atoms.calc.name == "vasp"
        ):
            warn(
                "Assuming Atoms object Singlepoint labeled 'vasp' is precalculated (to turn this behavior off: set 'reverify_with_parent' to True)"
            )
            # check if parent calc is a delta, if so: turn the vasp singlepoint into a deltacalc singlepoint
            if type(self.parent_calc) is DeltaCalc:
                (new_data,) = subtract_deltas(
                    [atoms], self.parent_calc.calcs[1], self.parent_calc.refs
                )
            # else just use the atoms as normal
            else:
                new_data = atoms
        # if verifying (or reverifying) do the singlepoints
        else:
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

        if (
            len(self.parent_dataset)
            and self.learner_params.get("partial_fit", False) >= 2
        ):
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
