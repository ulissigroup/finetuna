from al_mlp.utils import compute_with_calc
from al_mlp.offline_active_learner import OfflineActiveLearner
import numpy as np

class TestLearner(OfflineActiveLearner):
    """
    Replaces termination criteria with a max force in the constructor and the check_terminate method
    """

    def __init__(
        self, learner_params, trainer, training_data, parent_calc, base_calc
    ):
        super().__init__(learner_params, trainer, training_data, parent_calc, base_calc)
        self.max_evA = learner_params["max_evA"]

    def check_terminate(self):
        """
        Terminate when forces get below max optimum force on final DFT call in trajectory
        """
        final_point_image = [self.sample_candidates[-1]]
        final_point_evA = compute_with_calc(final_point_image, self.parent_calc)
        if np.max(np.abs(final_point_evA[0].get_forces())) <= self.max_evA:
            return True
        return False