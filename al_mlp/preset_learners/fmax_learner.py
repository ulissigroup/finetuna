from al_mlp.offline_learner import OfflineActiveLearner
from al_mlp.utils import compute_with_calc, write_to_db
import numpy as np
import ase
import random


class FmaxLearner(OfflineActiveLearner):
    """
    Replaces termination criteria with a max force in the constructor and the check_terminate method
    """

    def __init__(self, learner_params, trainer, training_data, parent_calc, base_calc):
        super().__init__(learner_params, trainer, training_data, parent_calc, base_calc)
        self.max_evA = learner_params["max_evA"]

    def check_terminate(self):
        """
        Default termination function.
        """
        if self.iterations >= self.max_iterations:
            return True
        else:
            if self.iterations > 0 and self.final_point_force <= self.max_evA:
                return True
        return False

    def query_func(self):
        """
        Default random query strategy.
        """
        queries_db = ase.db.connect("queried_images.db")
        query_idx = random.sample(
            range(1, len(self.sample_candidates) - 1),
            self.samples_to_retrain - 1,
        )
        queried_images = [self.sample_candidates[idx] for idx in query_idx]
        query_idx = np.append(query_idx, [len(self.sample_candidates) - 1])
        write_to_db(queries_db, queried_images)
        self.parent_calls += len(queried_images)
        return queried_images

    def query_data(self):
        """
        Queries data from a list of images. Calculates the properties
        and adds them to the training data.
        """
        final_point_image = [self.sample_candidates[-1]]
        final_point_evA = compute_with_calc(final_point_image, self.parent_calc)
        self.parent_calls += 1
        self.training_data += compute_with_calc(final_point_image, self.delta_sub_calc)
        self.final_point_force = np.max(np.abs(final_point_evA[0].get_forces()))

        random.seed(self.query_seeds[self.iterations - 1])
        queried_images = self.query_func()
        self.training_data += compute_with_calc(queried_images, self.delta_sub_calc)


class ForceQueryLearner(FmaxLearner):
    """
    Terminates based on max force.
    Guarantees the query of the image with the lowest ML fmax.
    """

    def query_func(self):
        """
        Queries the minimum fmax image + random
        """
        fmaxes = [
            np.max(np.abs(image.get_forces())) for image in self.sample_candidates[1:]
        ]
        min_index = np.argmin(fmaxes) + 1
        idxs = set(range(1, len(self.sample_candidates)))
        idxs.remove(min_index)

        queries_db = ase.db.connect("queried_images.db")
        query_idx = random.sample(idxs, self.samples_to_retrain - 1)
        query_idx.append(min_index)
        queried_images = [self.sample_candidates[idx] for idx in query_idx]
        write_to_db(queries_db, queried_images)
        self.parent_calls += len(queried_images)
        return queried_images

    def query_data(self):
        """
        Queries data from a list of images. Calculates the properties
        and adds them to the training data.
        """

        random.seed(self.query_seeds[self.iterations - 1])
        queried_images = self.query_func()
        self.training_data += compute_with_calc(queried_images, self.delta_sub_calc)
