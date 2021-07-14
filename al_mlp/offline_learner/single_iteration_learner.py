from al_mlp.offline_learner.offline_learner import OfflineActiveLearner


class SingleIterationLearner(OfflineActiveLearner):
    """Offline Active Learner for training a single iteration.
    Parameters
    ----------
    learner_settings: dict
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

    def __init__(
        self,
        learner_params,
        trainer,
        training_data,
        parent_calc,
        base_calc,
        mongo_db=None,
    ):
        super().__init__(
            learner_params,
            trainer,
            training_data,
            parent_calc,
            base_calc,
            mongo_db=mongo_db,
        )

    def init_learner(self):
        """
        Initializes learner, before training loop.
        """
        global compute_with_calc
        if self.learner_params["use_dask"]:
            from al_mlp.utils_dask import compute_with_calc
        else:
            from al_mlp.utils import compute_with_calc

        self.terminate = False
        self.iterations = 0
        self.atomistic_method = self.learner_params["atomistic_method"]
        self.filename = self.learner_params["filename"]
        self.file_dir = self.learner_params["file_dir"]

    def check_terminate(self):
        return True
