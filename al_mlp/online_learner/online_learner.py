import copy
import numpy as np
from ase.calculators.calculator import Calculator
from al_mlp.utils import convert_to_singlepoint
import pymongo
from atomate.vasp.database import VaspCalcDb
import datetime
from al_mlp.mongo import make_doc_from_atoms
import hashlib

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
        launch_id
    ):
        Calculator.__init__(self)

        self.parent_calc = parent_calc
        self.trainer = trainer
        self.learner_params = learner_params
        self.parent_dataset = convert_to_singlepoint(parent_dataset)

        self.ml_potential = ml_potential
        self.task_name = task_name
        self.launch_id = launch_id

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
        self.iteration = 0

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        # breakpoint()
        # If can make a connection to MongoDB then create a database to house the OAL outputs
        conn = self.mongodb_conn()

        if conn is not None:
            # Look to see if the oal_metadata collection exists and if not create it
            db = conn.db

            if "oal_metadata" not in db.collection_names():
                db.create_collection("oal_metadata")
            if "atoms_objects" not in db.collection_names():
                db.create_collection("atoms_objects")

            collection = db.get_collection("oal_metadata")

        # If we have less than two data points, uncertainty is not
        # well calibrated so just use DFT

        if len(self.parent_dataset) < 2:
            energy, force = self.add_data_and_retrain(atoms)
            self.results["energy"] = energy
            self.results["forces"] = force

            if conn is not None:
                self.insert_row(
                    collection,
                    iteration=self.iteration,
                    parent_call=True,
                    energy=energy,
                    force=force.tolist(),
                    max_force_stds="NA",
                    base_uncertainty="NA",
                    uncertainty_tol="NA",
                    max_force=float(np.nanmax(np.abs(force))), # log forces with constraints since they are used for convergence criteria
                    task_name=self.task_name,
                    launch_id=self.launch_id,
                    time_stamp=datetime.datetime.utcnow(),
                )
                self.insert_atoms_object(atoms, db)
                

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
            energy = atoms_ML.get_potential_energy()
            force = atoms_ML.get_forces()

        # Log to the database the metadata for the step

        if conn is not None:
            self.insert_row(
                collection,
                iteration=self.iteration,
                parent_call=parent_call,
                energy=energy,
                force=force.tolist(),
                max_force_stds=float(self.uncertainty),
                base_uncertainty=float(self.base_uncertainty),
                uncertainty_tol=float(self.uncertainty_tol),
                max_force=float(np.nanmax(np.abs(force))),
                task_name=self.task_name,
                launch_id=self.launch_id,
                time_stamp=datetime.datetime.utcnow(),
            )
            self.insert_atoms_object(atoms, db)

        self.iteration += 1
        # Return the energy/force
        self.results["energy"] = energy
        self.results["forces"] = force

    def unsafe_prediction(self, atoms):
        # Set the desired tolerance based on the current max predcited force
        self.uncertainty = atoms.calc.results["max_force_stds"]
        self.base_uncertainty = np.nanmax(np.abs(atoms.get_forces()))
        self.uncertainty_tol = max(
            [self.dyn_uncertain_tol * self.base_uncertainty, self.stat_uncertain_tol]
        )

        print(
            "Max Force Std: %1.3f eV/A, Max Force Threshold: %1.3f eV/A"
            % (self.uncertainty, self.uncertainty_tol)
        )

        print(
            "static tol: %1.3f eV/A, dynamic tol: %1.3f eV/A"
            % (self.stat_uncertain_tol, self.dyn_uncertain_tol * self.base_uncertainty)
        )
        if self.uncertainty > self.uncertainty_tol:
            maxf = np.nanmax(np.abs(atoms.get_forces()))
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
        atoms_copy.set_calculator(self.parent_calc)
        print(atoms_copy)
        (new_data,) = convert_to_singlepoint([atoms_copy])

        energy_actual = new_data.get_potential_energy()
        force_actual = new_data.get_forces()

        self.parent_dataset += [new_data]

        # Don't bother training if we have less than two datapoints

        if len(self.parent_dataset) >= 2:
            self.ml_potential.train(self.parent_dataset)

        self.parent_calls += 1

        return energy_actual, force_actual

    @classmethod
    def mongodb_conn(cls):
        """Checks if we can make a connection to a database and if yes returns the connection"""
        db = VaspCalcDb.from_db_file('/home/jovyan/atomate/config/db.json') # hardcoded to the nersc DB for now
        return db

    def insert_row(self, collection, **kw_args):
        collection.insert_one(kw_args)

    def insert_atoms_object(self, atoms, db):
        # insert the Atoms object into another collection
        doc = make_doc_from_atoms(atoms)
        # check if the doc already exists in the collection using hash field
        string = str(doc)
        hash_ = hashlib.md5(string.encode('utf-8')).hexdigest()
        if db["atoms_objects"].find_one({'hash_': hash_}) is None:
            doc['hash_'] = hash_
            self.insert_row(db["atoms_objects"], **doc)






