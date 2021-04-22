from al_mlp.offline_learner import OfflineActiveLearner
from al_mlp.utils import compute_with_calc, write_to_db, subtract_deltas
import numpy as np
import ase
import random
from al_mlp.calcs import DeltaCalc


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
        if len(self.sample_candidates) <= self.samples_to_retrain:
            print(
                "Number of sample candidates is less than or equal to the requested samples to retrain, defaulting to all samples but the initial and final"
            )
            query_idx = [*range(1, len(self.sample_candidates) - 1)]
            if query_idx == []:
                query_idx = [
                    1
                ]  # EDGE CASE WHEN samples = 2 (need a better way to do it)

        else:
            query_idx = random.sample(
                range(1, len(self.sample_candidates) - 1),
                self.samples_to_retrain - 1,
            )
        # query_idx = np.append(query_idx, [len(self.sample_candidates) - 1])
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

        final_point_image = [self.sample_candidates[-1]]
        # print(final_point_image[0].get_positions())
        final_point_evA = compute_with_calc(final_point_image, self.parent_calc)
        self.final_point_force = np.max(np.abs(final_point_evA[0].get_forces()))
        self.training_data += subtract_deltas(
            final_point_evA, self.base_calc, self.refs
        )
        self.parent_calls += 1
        queries_db = ase.db.connect("queried_images.db")
        random.seed(self.query_seeds[self.iterations - 1] + 1)
        write_to_db(queries_db, final_point_image)

        self.terminate = self.check_terminate()
        self.iterations += 1


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
        queried_images = [self.sample_candidates[idx] for idx in query_idx]
        min_force_image = self.sample_candidates[min_index]
        write_to_db(queries_db, queried_images)
        write_to_db(queries_db, [min_force_image])
        self.parent_calls += len(queried_images) + 1
        return queried_images, min_force_image

    def query_data(self):
        """
        Queries data from a list of images. Calculates the properties
        and adds them to the training data.
        """

        random.seed(self.query_seeds[self.iterations - 1])
        random_queried_images, min_force_image = self.query_func()
        self.training_data += compute_with_calc(
            random_queried_images, self.delta_sub_calc
        )
        min_image_parent = compute_with_calc([min_force_image], self.parent_calc)[0]
        self.final_point_force = np.max(np.abs(min_image_parent.get_forces()))
        self.training_data += subtract_deltas(
            [min_image_parent], self.base_calc, self.refs
        )
