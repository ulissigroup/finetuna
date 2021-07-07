import copy
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np
from ase.calculators.calculator import Calculator
from al_mlp.utils import convert_to_singlepoint
from ase.calculators.singlepoint import SinglePointCalculator
import pymongo
from atomate.vasp.database import VaspCalcDb
import datetime
from al_mlp.mongo import make_doc_from_atoms, make_atoms_from_doc
from pymongo.errors import InvalidDocument
from deepdiff import DeepHash
from copy import deepcopy

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
        task_name,
        launch_id,
        fw_id
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
        self.task_name = task_name
        self.launch_id = launch_id
        self.fw_id = fw_id

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
            energy, force, atoms_sp, force_cons = self.add_data_and_retrain(atoms)
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
                    max_force=float(np.sqrt((force ** 2).sum(axis=1).max())),
                    max_force_cons=float(np.sqrt((force_cons ** 2).sum(axis=1).max())),
                    task_name=self.task_name,
                    launch_id=self.launch_id,
                    time_stamp=datetime.datetime.utcnow(),
                )
                self.insert_atoms_object(atoms_sp, db)
                

            self.iteration += 1

            return

        # Make a copy of the atoms with ensemble energies as a SP
        atoms_ML = atoms.copy()
        atoms_ML.set_calculator(self.ml_potential)
        atoms_ML.get_forces(apply_constraint=False)

        #(atoms_ML_sp,) = convert_to_singlepoint([atoms_ML])

        # Check if we are extrapolating too far, and if so add/retrain

        # Initialize the parent_call boolean variable
        parent_call = False

        if self.unsafe_prediction(atoms_ML) or self.parent_verify(atoms_ML):
            parent_call = True
            # We ran DFT, so just use that energy/force
            energy, force, atoms_sp, force_cons = self.add_data_and_retrain(atoms)
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
                    max_force=float(np.sqrt((force ** 2).sum(axis=1).max())),
                    max_force_cons=float(np.sqrt((force_cons ** 2).sum(axis=1).max())),
                    max_force_oal_cons=float(np.sqrt((atoms_ML.get_forces() ** 2).sum(axis=1).max())),
                    task_name=self.task_name,
                    launch_id=self.launch_id,
                    time_stamp=datetime.datetime.utcnow(),
                )
                self.insert_atoms_object(atoms_sp, db)
        else:
            energy = atoms_ML.get_potential_energy(apply_constraint=False)
            force = atoms_ML.get_forces(apply_constraint=False)
            atoms_ML_copy = atoms_ML.copy()
            calc = SinglePointCalculator(energy=energy,
                                         forces=force,
                                         atoms=atoms_ML_copy)
            atoms_ML_copy.set_calculator(calc)
            force_cons = atoms_ML.get_forces()
            if conn is not None:
                try:
                    self.insert_row(
                        collection,
                        iteration=self.iteration,
                        parent_call=parent_call,
                        energy=energy,
                        force=force.tolist(),
                        max_force_stds=float(self.uncertainty),
                        base_uncertainty=float(self.base_uncertainty),
                        uncertainty_tol=float(self.uncertainty_tol),
                        max_force=float(np.sqrt((force ** 2).sum(axis=1).max())),
                        max_force_cons=float(np.sqrt((force_cons ** 2).sum(axis=1).max())),
                        task_name=self.task_name,
                        launch_id=self.launch_id,
                        time_stamp=datetime.datetime.utcnow(),
                    )
                    self.insert_atoms_object(atoms_ML_copy, db)
                except InvalidDocument as e:
                    print(f"Failed to insert Atoms object because of {e}")
                    pass

        # Log to the database the metadata for the step


        self.iteration += 1
        # Return the energy/force
        self.results["energy"] = energy
        self.results["forces"] = force

    def unsafe_prediction(self, atoms):
        # Set the desired tolerance based on the current max predcited force
        self.uncertainty = atoms.calc.results["max_force_stds"]
        forces = atoms.get_forces(apply_constraint=False)
        self.base_uncertainty = np.sqrt((forces ** 2).sum(axis=1).max())
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
            return True
        else:
            return False

    def parent_verify(self, atoms):
        forces = atoms.get_forces()
        fmax = np.sqrt((forces ** 2).sum(axis=1).max())
        return fmax <= self.fmax_verify_threshold

    def add_data_and_retrain(self, atoms):
        print("OnlineLearner: Parent calculation required")
        if (
            self.max_parent_calls is not None
            and self.parent_calls >= self.max_parent_calls
        ):
            print("Parent call failed: max parent calls reached")
            atoms.set_calculator(self.ml_potential)
            energy = atoms.get_potential_energy(apply_constraint=False)
            force = atoms.get_forces(apply_constraint=False)
            force_cons = atoms.get_forces()
            return energy, force, force_cons

        start = time.time()

        atoms_copy = atoms.copy() # YURI Let the atoms get mutated so that we can store it with all the attributes in the DB
        atoms_copy.set_calculator(self.parent_calc)
        print(atoms_copy)
        (new_data,) = convert_to_singlepoint([atoms_copy]) # Where DFT happens

        energy_actual = new_data.get_potential_energy(apply_constraint=False)
        force_actual = new_data.get_forces(apply_constraint=False)
        force_cons = new_data.get_forces()

        conn = self.mongodb_conn()

        if conn is not None:
            db = conn.db
        parent_data = [make_atoms_from_doc(doc) for doc in db['atoms_objects'].find({})]
        self.parent_dataset = parent_data

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

        return energy_actual, force_actual, new_data, force_cons

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
        # Make a copy of the doc in which we pop the 'user', 'mtime','ctime' keys to exclude from hash
        doc_copy = deepcopy(doc)
        for key in ('user', 'mtime', 'ctime'):
            doc_copy.pop(key)
        # check if the doc already exists in the collection using hash field
        hash_ = DeepHash(doc_copy, number_format_notation="e", significant_digits=3)[doc_copy] # Use scientific notation and only look
        del doc_copy # Dispose of the doc_copy
        # upto three digits after the decimal to see if float should map to the same hash
        if db["atoms_objects"].find_one({'hash_': hash_}) is None:
            doc['hash_'] = hash_
            doc['launch_id'] = self.launch_id
            doc['fw_id'] = self.fw_id
            self.insert_row(db["atoms_objects"], **doc)
        else:
            print('Duplicate found, not adding to DB...')






