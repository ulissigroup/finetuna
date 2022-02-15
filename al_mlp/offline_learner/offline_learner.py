import random
from ase.io.trajectory import Trajectory

from ase.optimize.bfgs import BFGS
from al_mlp.atomistic_methods import Relaxation
from al_mlp.base_calcs.dummy import Dummy
from al_mlp.calcs import DeltaCalc
from al_mlp.utils import compute_with_calc
import numpy as np
from ase.calculators.calculator import Calculator
from al_mlp.logger import Logger


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
        optional_config=None,
    ):
        self.learner_params = learner_params
        self.ml_potential = ml_potential
        self.training_data = training_data
        self.parent_calc = parent_calc
        self.base_calc = base_calc
        self.calcs = [parent_calc, base_calc]

        if mongo_db is None:
            mongo_db = {"offline_learner": None}
        self.logger = Logger(
            learner_params=learner_params,
            ml_potential=ml_potential,
            parent_calc=parent_calc,
            base_calc=base_calc,
            mongo_db_collection=mongo_db["offline_learner"],
            optional_config=optional_config,
        )

        self.init_learner()
        self.init_training_data()

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

        substep = 0
        for image in self.sample_candidates:
            energy = image.get_potential_energy(apply_constraint=False)
            forces = image.get_forces(apply_constraint=False)
            constrained_forces = image.get_forces()
            fmax = np.sqrt((constrained_forces**2).sum(axis=1).max())
            info = {
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
            self.logger.write(image, info)

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

        for i in range(len(un_delta_new_dataset)):
            image = un_delta_new_dataset[i]
            idx = None
            if query_idx is not None:
                idx = query_idx[i]
            energy = image.get_potential_energy(apply_constraint=False)
            forces = image.get_forces(apply_constraint=False)
            constrained_forces = image.get_forces()
            fmax = np.sqrt((constrained_forces**2).sum(axis=1).max())
            info = {
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
            self.logger.write(image, info)

        return un_delta_new_dataset

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
