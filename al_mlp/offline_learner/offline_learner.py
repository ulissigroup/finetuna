import random
from ase.io.trajectory import Trajectory

from ase.optimize.bfgs import BFGS
from al_mlp.atomistic_methods import Relaxation
from al_mlp.base_calcs.dummy import Dummy
from al_mlp.utils import compute_with_calc
import numpy as np
from ase.calculators.calculator import Calculator
from al_mlp.logger import Logger


class OfflineLearner:
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

    """

    def __init__(
        self,
        learner_params,
        training_data,
        ml_potential,
        parent_calc,
        mongo_db=None,
        optional_config=None,
    ):
        self.learner_params = learner_params
        self.ml_potential = ml_potential
        self.initial_data = training_data
        self.parent_calc = parent_calc

        self.init_logger(mongo_db, optional_config)
        self.init_learner()
        self.init_training_data()

    def init_logger(self, mongo_db, optional_config):
        if mongo_db is None:
            mongo_db = {"offline_learner": None}
        self.logger = Logger(
            learner_params=self.learner_params,
            ml_potential=self.ml_potential,
            parent_calc=self.parent_calc,
            mongo_db_collection=mongo_db["offline_learner"],
            optional_config=optional_config,
        )

    def init_learner(self):
        """
        Initializes learner, before training loop.
        """

        self.iterations = 0
        self.parent_calls = 0
        self.terminate = False

        atomistic_method = self.learner_params.get("atomistic_method")
        if type(atomistic_method) is Relaxation:
            self.atomistic_method = atomistic_method
        elif type(atomistic_method) is dict:
            self.atomistic_method = Relaxation(
                initial_geometry=Trajectory(
                    self.learner_params.get("atomistic_method").get("initial_traj")
                )[0],
                optimizer=BFGS,
                fmax=self.learner_params.get("atomistic_method", {}).get("fmax", 0.03),
                steps=self.learner_params.get("atomistic_method", {}).get(
                    "steps", 2000
                ),
                maxstep=self.learner_params.get("atomistic_method", {}).get(
                    "maxstep", 0.04
                ),
            )
        else:
            raise TypeError(
                "Passed in config without an atomistic method Relaxation object or dictionary"
            )

        self.max_iterations = self.learner_params.get("max_iterations", 20)
        self.samples_to_retrain = self.learner_params.get("samples_to_retrain", 1)
        self.filename = self.learner_params.get("filename", "relax_example")
        self.file_dir = self.learner_params.get("file_dir", "./")
        self.seed = self.learner_params.get("seed", random.randint(0, 100000))

        random.seed(self.seed)
        self.query_seeds = random.sample(range(100000), self.max_iterations)

    def init_training_data(self):
        """
        Prepare the training data and the model for the training loop.
        Run an initial relaxation.
        """
        self.training_data = []

        self.add_data(self.initial_data, None)
        self.do_train()
        self.do_after_train()

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

        Queries data from a list of images,
        calculates the properties,
        adds them to the training data,
        then logs them as parent data.
        """

        random.seed(self.query_seeds[self.iterations - 1])
        queried_images, query_idx = self.query_func()
        self.add_data(queried_images, query_idx)

    def do_train(self):
        """
        Executes the training of ml_potential
        """
        self.ml_potential.train(self.training_data)

    def do_after_train(self):
        """
        Executes after training the ml_potential in every active learning loop.
        """
        trained_calc = self.make_trainer_calc()

        self.fn_label = f"{self.file_dir}{self.filename}_iter_{self.iterations}"
        self.atomistic_method.run(calc=trained_calc, filename=self.fn_label)
        self.sample_candidates = list(
            self.atomistic_method.get_trajectory(filename=self.fn_label)
        )

        self.log_ml_data()
        self.terminate = self.check_terminate()
        self.iterations += 1

    def do_after_learn(self):
        """
        Executes after active learning loop terminates.
        """
        pass

    def add_data(self, queried_images, query_idx):
        """
        Attaches calculators to the queried images and runs calculate
        Adds the training-ready images to the training dataset
        Returns the  images in a list called new_dataset
        """
        new_dataset = compute_with_calc(queried_images, self.make_trainer_calc())
        self.training_data += new_dataset
        self.parent_calls += len(new_dataset)
        self.log_parent_data(new_dataset, query_idx)

        return new_dataset

    def log_parent_data(self, new_dataset, query_idx):
        """
        Gets the energy and forces of the new dataset
        Logs all parameters with logger
        """
        for i, image in enumerate(new_dataset):
            idx = None
            if query_idx is not None:
                idx = query_idx[i]
            energy = image.get_potential_energy(apply_constraint=False)
            forces = image.get_forces(apply_constraint=False)
            constrained_forces = image.get_forces()
            fmax = np.sqrt((constrained_forces ** 2).sum(axis=1).max())
            self.info = {
                "check": True,
                "energy": energy,
                "forces": forces,
                "fmax": fmax,
                "ml_energy": None,
                "ml_forces": None,
                "ml_fmax": None,
                "parent_energy": energy,
                "parent_forces": forces,
                "parent_fmax": fmax,
                "force_uncertainty": image.info.get("max_force_stds", None),
                "energy_uncertainty": image.info.get("energy_stds", None),
                "dyn_uncertainty_tol": None,
                "stat_uncertain_tol": None,
                "tolerance": None,
                "parent_calls": self.parent_calls,
                "trained_on": True,
                "query_idx": idx,
                "substep": idx,
            }
            self.logger.write(image, self.info)

    def log_ml_data(self):
        substep = 0
        for image in self.sample_candidates:
            energy = image.get_potential_energy(apply_constraint=False)
            forces = image.get_forces(apply_constraint=False)
            constrained_forces = image.get_forces()
            fmax = np.sqrt((constrained_forces ** 2).sum(axis=1).max())
            self.info = {
                "check": False,
                "energy": energy,
                "forces": forces,
                "fmax": fmax,
                "ml_energy": energy,
                "ml_forces": forces,
                "ml_fmax": fmax,
                "parent_energy": None,
                "parent_forces": None,
                "parent_fmax": None,
                "force_uncertainty": image.info.get("max_force_stds", None),
                "energy_uncertainty": image.info.get("energy_stds", None),
                "dyn_uncertainty_tol": None,
                "stat_uncertain_tol": None,
                "tolerance": None,
                "parent_calls": self.parent_calls,
                "trained_on": False,
                "query_idx": None,
                "substep": substep,
            }
            substep += 1
            self.logger.write(image, self.info)

    def check_terminate(self):
        """
        Default termination function.
        """
        final_image = self.sample_candidates[-1]
        query_idx = len(self.sample_candidates) - 1
        final_image = self.add_data([final_image], [query_idx])[0]
        max_force = np.sqrt((final_image.get_forces() ** 2).sum(axis=1).max())
        terminate = False
        if max_force <= self.atomistic_method.fmax:
            terminate = True
        print(
            "Final image check with parent calc: "
            + str(terminate)
            + ", energy: "
            + str(final_image.get_potential_energy())
            + ", max force: "
            + str(max_force)
        )
        if self.iterations >= self.max_iterations:
            return True
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
