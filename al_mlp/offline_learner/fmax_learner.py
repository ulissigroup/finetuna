from al_mlp.offline_learner.offline_learner import OfflineActiveLearner
from al_mlp.utils import compute_with_calc, write_to_db
import numpy as np
import ase
import random
from ase.io.trajectory import TrajectoryWriter


class FmaxLearner(OfflineActiveLearner):
    """
    Replaces termination criteria with a max force in the constructor and the check_terminate method
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
        super().__init__(
            learner_params,
            training_data,
            ml_potential,
            parent_calc,
            base_calc,
            mongo_db=mongo_db,
        )
        self.max_evA = learner_params["max_evA"]

    def check_terminate(self):
        """
        Termination function.
        """
        final_point_image = [self.sample_candidates[-1]]
        final_point_evA = compute_with_calc(final_point_image, self.delta_sub_calc)
        self.final_point_force = final_point_evA[0].info["parent fmax"]
        self.training_data += final_point_evA
        self.parent_calls += 1
        random.seed(self.query_seeds[self.iterations - 1] + 1)

        if self.iterations == 0:
            writer = TrajectoryWriter("final_images.traj", mode="w")
            writer.write(final_point_image[0])
        else:
            writer = TrajectoryWriter("final_images.traj", mode="a")
            writer.write(final_point_image[0])

        if self.iterations >= self.max_iterations:
            return True
        else:
            if self.iterations > 0 and self.final_point_force <= self.max_evA:
                return True
        return False

    def query_func(self):
        """
        Random query strategy.
        """
        # queries_db = ase.db.connect("queried_images.db")
        if len(self.sample_candidates) <= self.samples_to_retrain:
            print(
                "Number of sample candidates is less than or equal to the requested samples to retrain, defaulting to all samples but the initial and final"
            )
            self.query_idx = [*range(1, len(self.sample_candidates) - 1)]
            if self.query_idx == []:
                self.query_idx = [
                    0
                ]  # EDGE CASE WHEN samples = 2 (need a better way to do it)

        else:
            self.query_idx = random.sample(
                range(1, len(self.sample_candidates) - 1),
                self.samples_to_retrain - 1,
            )
        queried_images = [self.sample_candidates[idx] for idx in self.query_idx]
        if self.iterations == 1:
            writer = TrajectoryWriter("queried_images.traj", mode="w")
            for i in queried_images:
                writer.write(i)
        else:
            writer = TrajectoryWriter("queried_images.traj", mode="a")
            for i in queried_images:
                writer.write(i)

        self.parent_calls += len(queried_images)
        return queried_images


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
        self.query_idx = random.sample(idxs, self.samples_to_retrain - 1)
        queried_images = [self.sample_candidates[idx] for idx in self.query_idx]
        min_force_image = self.sample_candidates[min_index]
        queried_images += min_force_image
        min_image_parent = compute_with_calc([min_force_image], self.parent_calc)[0]
        self.final_point_force = np.sqrt(
            (min_image_parent.get_forces() ** 2).sum(axis=1).max()
        )
        write_to_db(queries_db, queried_images)
        return queried_images
