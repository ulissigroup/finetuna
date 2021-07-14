from al_mlp.offline_learner.offline_learner import OfflineActiveLearner

# Specific NEB querying strategy


class NEBLearner(OfflineActiveLearner):
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
        self.parent_calls = 0

    def check_terminate(self):
        """
        Default termination function.
        """
        if self.iterations >= self.max_iterations:
            return True
        return False

    def query_func(self):
        """
        NEB query strategy.
        """
        # queries_db = ase.db.connect("queried_images.db")
        query_idx = [0, 2, 4]
        queried_images = [self.sample_candidates[idx] for idx in query_idx]
        self.parent_calls += len(queried_images)
        # write_to_db(queries_db,queried_images)
        return queried_images
