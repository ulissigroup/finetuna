import random
from al_mlp.base_calcs.dummy import Dummy
from al_mlp.calcs import DeltaCalc
from al_mlp.utils import convert_to_singlepoint, compute_with_calc, write_to_db
import ase
from al_mlp.mongo import MongoWrapper
import numpy as np
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator as sp


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

        self.init_learner()
        self.init_training_data()

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
        # setup delta sub calc as defacto parent calc for all queries
        parent_ref_image = self.atomistic_method.initial_geometry
        base_ref_image = compute_with_calc([parent_ref_image], self.base_calc)[0]
        self.refs = [parent_ref_image, base_ref_image]
        self.delta_sub_calc = DeltaCalc(self.calcs, "sub", self.refs)

        # move training data into raw data for computing with delta calc
        raw_data = []
        for image in self.training_data:
            raw_data.append(image)

        # run a trajectory with no training data: just the base model to sample from
        self.training_data = []
        self.fn_label = f"{self.file_dir}{self.filename}_iter_{self.iterations}"
        self.do_after_train()

        # add initial data to training dataset
        self.add_data(raw_data, None)
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
        queried_images, query_idx = self.query_func()
        self.add_data(queried_images, query_idx)

    def add_data(self, queried_images, query_idx):
        self.new_dataset = compute_with_calc(queried_images, self.delta_sub_calc)
        self.training_data += self.new_dataset
        self.parent_calls += len(self.new_dataset)

        un_delta_new_dataset = []
        for image in self.new_dataset:
            add_delta_calc = DeltaCalc([image.calc, self.base_calc], "add", self.refs)
            [un_delta_image] = compute_with_calc([image], add_delta_calc)
            un_delta_new_dataset.append(un_delta_image)

        if query_idx is None:
            tag = "initial"
        else:
            tag = "queried"
        queries_db = ase.db.connect("queried_images.db")
        for image in un_delta_new_dataset:
            parent_E = image.info["parent energy"]
            base_E = image.info["base energy"]
            write_to_db(queries_db, [image], tag, parent_E, base_E)
        self.write_to_mongo(
            check=True,
            list_of_atoms=un_delta_new_dataset,
            query_idx=query_idx,
            trained_on=True,
        )
        return un_delta_new_dataset

    def check_terminate(self):
        """
        Default termination function.
        """
        if self.iterations >= self.max_iterations:
            return True
        final_image = self.sample_candidates[-1]
        query_idx = [len(self.sample_candidates) - 1]
        final_image = self.add_data([final_image], [query_idx])[0]
        max_force = np.sqrt((final_image.get_forces() ** 2).sum(axis=1).max())
        terminate = False
        if max_force <= self.learner_params["atomistic_method"].fmax:
            terminate = True
        print(
            "Final image check with parent calc: "
            + str(terminate)
            + ", energy: "
            + str(final_image.get_potential_energy())
            + ", max force: "
            + str(max_force)
        )
        return terminate

    def query_func(self):
        """
        Default random query strategy.
        """
        if self.samples_to_retrain < 2 and self.training_data == 0:
            query_idx = random.sample(
                range(1, len(self.sample_candidates)),
                2,
            )
        else:
            query_idx = random.sample(
                range(1, len(self.sample_candidates)),
                self.samples_to_retrain,
            )
        queried_images = [self.sample_candidates[idx] for idx in query_idx]
        return queried_images, query_idx

    def make_trainer_calc(self, ml_potential=None):
        """
        Default ml_potential calc after train. Assumes ml_potential has a 'get_calc'
        method.
        If ml_potential is passed in, it will get its calculator instead
        """
        if len(self.training_data) == 0:
            return Dummy()
        if ml_potential is None:
            ml_potential = self.ml_potential
        if not isinstance(ml_potential, Calculator):
            calc = ml_potential.get_calc()
        else:
            calc = ml_potential
        return calc

    def write_to_mongo(self, check, list_of_atoms, query_idx=None, trained_on=False):
        if self.mongo_wrapper is not None:
            for i in range(len(list_of_atoms)):
                image = list_of_atoms[i]
                info = {
                    "check": check,
                    "uncertainty": image.info["max_force_stds"],
                    "energy": image.get_potential_energy(),
                    "maxForce": np.sqrt((image.get_forces() ** 2).sum(axis=1).max()),
                    "forces": str(image.get_forces(apply_constraint=False)),
                    "query_idx": None,
                    "trained_on": trained_on,
                }

                if query_idx is not None:
                    info["query_idx"] = query_idx[i]
                if "force_stds" in image.calc.results:
                    info["force_stds"] = image.calc.results["force_stds"]
                self.mongo_wrapper.write_to_mongo(image, info)
