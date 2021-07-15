import random
from al_mlp.calcs import DeltaCalc
from al_mlp.utils import convert_to_singlepoint, compute_with_calc, write_to_db
import ase
from al_mlp.mongo import MongoWrapper
import numpy as np
from ase.calculators.calculator import Calculator


class OfflineActiveLearner:
    """Offline Active Learner.
    This class serves as a parent class to inherit more sophisticated
    learners with different query and termination strategies.

    Parameters
    ----------
    learner_params: dict
        Dictionary of learner parameters and settings.

    ml_potential: ase Calculator object
        An instance of an ml_potential calculator that has a train and predict method.

    training_data: list
        A list of ase.Atoms objects that have attached calculators.
        Used as the first set of training data.

    parent_calc: ase Calculator object
        Calculator used for querying training data.

    base_calc: ase Calculator object
        Calculator used to calculate delta data for training.

    """

    def __init__(
        self,
        learner_params,
        training_data,
        ml_potential,
        parent_calc,
        base_calc,
        mongo_db=None,
    ):
        self.learner_params = learner_params
        self.ml_potential = ml_potential
        self.training_data = training_data
        self.parent_calc = parent_calc
        self.base_calc = base_calc
        self.calcs = [parent_calc, base_calc]
        self.init_learner()
        self.init_training_data()
        if mongo_db is not None:
            self.mongo_wrapper = MongoWrapper(
                mongo_db["offline_learner"],
                learner_params,
                ml_potential,
                parent_calc,
                base_calc,
            )
        else:
            self.mongo_wrapper = None

    def init_learner(self):
        """
        Initializes learner, before training loop.
        """

        self.iterations = 0
        self.parent_calls = 0
        self.terminate = False
        self.atomistic_method = self.learner_params.get("atomistic_method")
        self.max_iterations = self.learner_params.get("max_iterations", 20)
        self.samples_to_retrain = self.learner_params.get("samples_to_retrain", 1)
        self.filename = self.learner_params.get("filename", "relax_example")
        self.file_dir = self.learner_params.get("file_dir", "./")
        self.seed = self.learner_params.get("seed", random.randint(0, 100000))

        random.seed(self.seed)
        self.query_seeds = random.sample(range(100000), self.max_iterations)
        ase.db.connect("queried_images.db", append=False)

    def init_training_data(self):
        """
        Prepare the training data by attaching delta values for training.
        """

        raw_data = self.training_data
        sp_raw_data = convert_to_singlepoint(raw_data)
        parent_ref_image = sp_raw_data[0]
        base_ref_image = compute_with_calc([parent_ref_image], self.base_calc)[0]
        self.refs = [parent_ref_image, base_ref_image]
        self.delta_sub_calc = DeltaCalc(self.calcs, "sub", self.refs)
        self.training_data = []
        queries_db = ase.db.connect("queried_images.db")
        for image in sp_raw_data:
            # sp_calc = image.get_calculator()
            # sp_delta_calc = DeltaCalc([sp_calc, self.base_calc], "sub", self.refs)
            sp_image = compute_with_calc([image], self.delta_sub_calc)
            self.training_data += sp_image
            parent_E = sp_image[0].info["parent energy"]
            base_E = sp_image[0].info["base energy"]
            write_to_db(queries_db, sp_image, "initial", parent_E, base_E)
        self.initial_image_energy = self.refs[0].get_potential_energy()

    def learn(self):
        """
        Conduct offline active learning.

        Parameters
        ----------
        atomistic_method: object
            Define relaxation parameters and starting image.
        """

        while not self.terminate:
            self.do_before_train()
            self.do_train()
            self.do_after_train()
        self.do_after_learn()

    def do_before_train(self):
        """
        Executes before training the ml_potential in every active learning loop.
        """
        if self.iterations > 0:
            self.query_data()
        self.fn_label = f"{self.file_dir}{self.filename}_iter_{self.iterations}"

    def do_train(self):
        """
        Executes the training of ml_potential
        """
        self.ml_potential.train(self.training_data)

    def do_after_train(self):
        """
        Executes after training the ml_potential in every active learning loop.
        """
        ml_potential = self.make_trainer_calc()
        self.trained_calc = DeltaCalc([ml_potential, self.base_calc], "add", self.refs)

        self.atomistic_method.run(calc=self.trained_calc, filename=self.fn_label)
        self.sample_candidates = list(
            self.atomistic_method.get_trajectory(filename=self.fn_label)
        )
        if self.mongo_wrapper is not None:
            self.mongo_wrapper.first = True
        self.write_to_mongo(check=False, list_of_atoms=self.sample_candidates)

        self.terminate = self.check_terminate()
        self.iterations += 1

    def do_after_learn(self):
        """
        Executes after active learning loop terminates.
        """
        pass

    def query_data(self):
        """
        Queries data from a list of images. Calculates the properties
        and adds them to the training data.
        """

        random.seed(self.query_seeds[self.iterations - 1])
        queried_images = self.query_func()
        self.new_dataset = compute_with_calc(queried_images, self.delta_sub_calc)
        self.training_data += self.new_dataset
        self.parent_calls += len(self.new_dataset)

        queries_db = ase.db.connect("queried_images.db")
        for image in self.new_dataset:
            parent_E = image.info["parent energy"]
            base_E = image.info["base energy"]
            write_to_db(queries_db, [image], "queried", parent_E, base_E)
        self.write_to_mongo(check=False, list_of_atoms=self.new_dataset)

    def check_terminate(self):
        """
        Default termination function.
        """
        if self.iterations >= self.max_iterations:
            return True
        return False

    def query_func(self):
        """
        Default random query strategy.
        """
        query_idx = random.sample(
            range(1, len(self.sample_candidates)),
            self.samples_to_retrain,
        )
        queried_images = [self.sample_candidates[idx] for idx in query_idx]
        return queried_images

    def make_trainer_calc(self, ml_potential=None):
        """
        Default ml_potential calc after train. Assumes ml_potential has a 'get_calc'
        method.
        If ml_potential is passed in, it will get its calculator instead
        """
        if ml_potential is not None:
            if not isinstance(ml_potential, Calculator):
                calc = ml_potential.get_calc()
            else:
                calc = ml_potential
        else:
            if not isinstance(self.ml_potential, Calculator):
                calc = self.ml_potential.get_calc()
            else:
                calc = self.ml_potential
        return calc

    def write_to_mongo(self, check, list_of_atoms):
        if self.mongo_wrapper is not None:
            for image in list_of_atoms:
                info = {
                    "check": check,
                    "uncertainty": image.info["max_force_stds"],
                    "energy": image.get_potential_energy(),
                    "maxForce": np.sqrt((image.get_forces() ** 2).sum(axis=1).max()),
                    "forces": str(image.get_forces(apply_constraint=False)),
                }
                self.mongo_wrapper.write_to_mongo(image, info)
