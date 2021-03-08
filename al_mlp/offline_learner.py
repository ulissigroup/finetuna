import random
from al_mlp.calcs import DeltaCalc
from al_mlp.utils import convert_to_singlepoint, compute_with_calc, write_to_db
import ase


class OfflineActiveLearner:
    """Offline Active Learner.
    This class serves as a parent class to inherit more sophisticated
    learners with different query and termination strategies.

    Parameters
    ----------
    learner_params: dict
        Dictionary of learner parameters and settings.

    trainer: object
        An isntance of a trainer that has a train and predict method.

    training_data: list
        A list of ase.Atoms objects that have attached calculators.
        Used as the first set of training data.

    parent_calc: ase Calculator object
        Calculator used for querying training data.

    base_calc: ase Calculator object
        Calculator used to calculate delta data for training.

    """

    def __init__(self, learner_params, trainer, training_data, parent_calc, base_calc):
        self.learner_params = learner_params
        self.trainer = trainer
        self.training_data = training_data
        self.parent_calc = parent_calc
        self.base_calc = base_calc
        self.calcs = [parent_calc, base_calc]
        self.init_learner()
        self.init_training_data()

    def init_learner(self):
        """
        Initializes learner, before training loop.
        """
        global compute_with_calc
        global convert_to_singlepoint
        if self.learner_params.get("use_dask", False):
            from al_mlp.dask_calculate import compute_with_calc
            from al_mlp.dask_calculate import convert_to_singlepoint

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
        for image in sp_raw_data:
            sp_calc = image.get_calculator()
            sp_calc.implemented_properties = ["energy", "forces"]
            sp_delta_calc = DeltaCalc([sp_calc, self.base_calc], "sub", self.refs)
            self.training_data += compute_with_calc([image], sp_delta_calc)

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
        Executes before training the trainer in every active learning loop.
        """
        if self.iterations > 0:
            self.query_data()
        self.fn_label = f"{self.file_dir}{self.filename}_iter_{self.iterations}"

    def do_train(self):
        """
        Executes the training of trainer
        """
        self.trainer.train(self.training_data)

    def do_after_train(self):
        """
        Executes after training the trainer in every active learning loop.
        """

        trainer_calc = self.make_trainer_calc()
        self.trained_calc = DeltaCalc([trainer_calc, self.base_calc], "add", self.refs)

        self.atomistic_method.run(calc=self.trained_calc, filename=self.fn_label)
        self.sample_candidates = list(
            self.atomistic_method.get_trajectory(filename=self.fn_label)
        )

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

        Parameters
        ----------
        sample_candidates: list
            List of ase atoms objects to query from.
        """

        random.seed(self.query_seeds[self.iterations - 1])
        queried_images = self.query_func()
        self.training_data += compute_with_calc(queried_images, self.delta_sub_calc)

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
        queries_db = ase.db.connect("queried_images.db")
        query_idx = random.sample(
            range(1, len(self.sample_candidates)),
            self.samples_to_retrain,
        )
        queried_images = [self.sample_candidates[idx] for idx in query_idx]
        write_to_db(queries_db, queried_images)
        self.parent_calls += len(queried_images)
        return queried_images

    def make_trainer_calc(self, trainer=None):
        """
        Default trainer calc after train. Assumes trainer has a 'get_calc'
        method.
        If trainer is passed in, it will get its calculator instead
        """
        if trainer is not None:
            return trainer.get_calc()
        return self.trainer.get_calc()
