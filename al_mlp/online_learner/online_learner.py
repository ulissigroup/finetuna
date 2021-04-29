import copy
import numpy as np
from ase.calculators.calculator import Calculator
from al_mlp.utils import convert_to_singlepoint
import pymongo
import datetime

__author__ = "Muhammed Shuaibi"
__email__ = "mshuaibi@andrew.cmu.edu"

class OnlineLearner(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        learner_params,
        trainer,
        parent_dataset,
        ml_potential,
        parent_calc,
        task_name,
    ):
        Calculator.__init__(self)

        self.parent_calc = parent_calc
        self.trainer = trainer
        self.learner_params = learner_params
        self.parent_dataset = convert_to_singlepoint(parent_dataset)

        self.ml_potential = ml_potential
        self.task_name = task_name

        # Don't bother training with only one data point,
        # as the uncertainty is meaningless

        if len(self.parent_dataset) > 1:
            ml_potential.train(self.parent_dataset)

        if "fmax_verify_threshold" in self.learner_params:
            self.fmax_verify_threshold = self.learner_params["fmax_verify_threshold"]
        else:
            self.fmax_verify_threshold = np.nan  # always False

        self.uncertain_tol = learner_params["uncertain_tol"]
        self.parent_calls = 0
        self.iteration = 0

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        # breakpoint()
        # If can make a connection to MongoDB then create a database to house the OAL outputs
        conn = self.mongodb_conn()

        if conn is not None:
            # Look to see if the oal_metadata collection exists and if not create it
            db = conn.get_database("db")

            if "oal_metadata" not in db.collection_names():
                db.create_collection("oal_metadata")

            collection = db.get_collection("oal_metadata")

        # If we have less than two data points, uncertainty is not
        # well calibrated so just use DFT

        if len(self.parent_dataset) < 2:
            energy, force = self.add_data_and_retrain(atoms)
            self.results["energy"] = energy
            self.results["forces"] = force

            if collection is not None:
                self.insert_row(
                    collection,
                    iteration=self.iteration,
                    parent_call=True,
                    energy=energy,
                    force=force.tolist(),
                    max_force_stds="NA",
                    base_uncertainty="NA",
                    uncertainty_tol="NA",
                    max_force=np.nanmax(np.abs(force)),
                    task_name=self.task_name,
                    time_stamp=datetime.datetime.utcnow(),
                )

            self.iteration += 1

            return

        # Make a copy of the atoms with ensemble energies as a SP
        atoms_ML = atoms.copy()
        atoms_ML.set_calculator(self.ml_potential)
        atoms_ML.get_forces()

        #         (atoms_ML,) = convert_to_singlepoint([atoms_copy])

        # Check if we are extrapolating too far, and if so add/retrain

        # Initialize the parent_call boolean variable
        parent_call = False

        if self.unsafe_prediction(atoms_ML) or self.parent_verify(atoms_ML):
            parent_call = True
            # We ran DFT, so just use that energy/force
            energy, force = self.add_data_and_retrain(atoms)
        else:
        #    if collection is not None:
            energy = atoms_ML.get_potential_energy(apply_constraint=False)
            force = atoms_ML.get_forces(apply_constraint=False)

        # Log to the database the metadata for the step

        if collection is not None:
            self.insert_row(
                collection,
                iteration=self.iteration,
                parent_call=parent_call,
                energy=energy,
                force=force.tolist(),
                max_force_stds=float(self.uncertainty),
                base_uncertainty=float(self.base_uncertainty),
                uncertainty_tol=float(self.uncertainty_tol),
                max_force=np.nanmax(np.abs(force)),
                task_name=self.task_name,
                time_stamp=datetime.datetime.utcnow(),
            )

        self.iteration += 1
        # Return the energy/force
        self.results["energy"] = energy
        self.results["forces"] = force

    def unsafe_prediction(self, atoms):
        # Set the desired tolerance based on the current max predcited force
        self.uncertainty = atoms.calc.results["max_force_stds"]
        #breakpoint()
        self.base_uncertainty = np.nanmax(np.abs(atoms.get_forces()))
        self.uncertainty_tol = self.uncertain_tol * self.base_uncertainty

        print(
            "Max Force Std: %1.3f eV/A, Max Force Threshold: %1.3f eV/A"
            % (self.uncertainty, self.uncertainty_tol)
        )

        if self.uncertainty > self.uncertainty_tol:
            return True
        else:
            return False

    def parent_verify(self, atoms):
        forces = atoms.get_forces()
        fmax = np.sqrt((forces ** 2).sum(axis=1).max())

        return fmax <= self.fmax_verify_threshold

    def add_data_and_retrain(self, atoms):
        print("OnlineLearner: Parent calculation required")

        atoms_copy = atoms.copy()
        atoms_copy.set_calculator(copy.copy(self.parent_calc))
        print(atoms_copy)
        (new_data,) = convert_to_singlepoint([atoms_copy])

        energy_actual = new_data.get_potential_energy(apply_constraint=False)
        force_actual = new_data.get_forces(apply_constraint=False)

        self.parent_dataset += [new_data]

        # Don't bother training if we have less than two datapoints

        if len(self.parent_dataset) >= 2:
            self.ml_potential.train(self.parent_dataset)

        self.parent_calls += 1

        return energy_actual, force_actual

    @classmethod
    def mongodb_conn(cls):
        """Checks if we can make a connection to a database and if yes returns the connection"""
        try:
            return pymongo.MongoClient()
        except pymongo.errors.ConnectionFailure as e:
            print(f"Could not connect to server: {e}")

    def insert_row(self, collection, **kw_args):
        collection.insert_one(kw_args)
