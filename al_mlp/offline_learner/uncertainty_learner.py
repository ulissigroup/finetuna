from al_mlp.offline_learner.ensemble_learner import EnsembleLearner
import ase
import random
from al_mlp.utils import write_to_db, compute_with_calc, subtract_deltas
import numpy as np
import matplotlib.pyplot as plt


class UncertaintyOffAL(EnsembleLearner):
    """
    Extends Ensemble Learner.
    Changes it to use a static uncertainty to restrict when to query
    Specifically, it will randomly query from within a given uncertainty threshold

    Can serve as a parent class for other uncertainty offline learners
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

    ensemble: int
        The number of models in ensemble

    unertainty_tol: float
        static threshold for uncertainty tolerance
    """

    def __init__(
        self,
        learner_params,
        trainer,
        training_data,
        parent_calc,
        base_calc,
        ml_potential,
        uncertainty_tol=0.5,
    ):
        super().__init__(
            learner_params,
            trainer,
            training_data,
            parent_calc,
            base_calc,
            ml_potential,
        )
        self.uncertainty_tol = uncertainty_tol
        self.max_evA = learner_params.get("max_evA", 0.05)
        self.final_point_force = 0
        self.rejected_list = {}

    def __str__(self):
        return "Default (Static) Uncertainty-Based Offline Learner"

    def restrict_candidates(self):
        restricted_candidates = []
        remaining_candidates = []
        uncertainty_list = []
        tol_list = []
        for i in range(len(self.sample_candidates)):
            uncertainty = self.sample_candidates[i].info["max_force_stds"]
            uncertainty_list.append(uncertainty)
            uncertainty_tol = (
                np.nanmax(np.abs(self.sample_candidates[i].get_forces()))
                * self.get_uncertainty_tol()
            )
            tol_list.append(uncertainty_tol)
            if uncertainty < uncertainty_tol:
                restricted_candidates.append(self.sample_candidates[i])

            else:
                remaining_candidates.append(self.sample_candidates[i])
        # if there aren't enough candidates based on criteria, get the lowest uncertainty remaining
        # candidates to return
        if len(restricted_candidates) < self.samples_to_retrain:
            print("Warning: not enough sample candidates which meet criteria")
            remaining_candidates.sort(
                key=lambda candidate: candidate.info["max_force_stds"]
            )
            restricted_candidates.extend(
                remaining_candidates[
                    : self.samples_to_retrain - len(restricted_candidates)
                ]
            )

        return restricted_candidates

    def check_final_force(self):
        final_point_image = [self.sample_candidates[-1]]
        final_point_evA = compute_with_calc(final_point_image, self.parent_calc)
        self.final_point_force = np.max(np.abs(final_point_evA[0].get_forces()))
        # self.energy_list.append(final_point_evA[0].get_potential_energy())
        final_point = subtract_deltas(final_point_evA, self.base_calc, self.refs)
        self.training_data += final_point
        # self.parent_dataset, self.training_data = self.add_data(final_point)
        self.parent_calls += 1

    def query_func(self):
        restricted_candidates = self.restrict_candidates()
        query_idx = self.sub_query_func(restricted_candidates)
        queried_images = [restricted_candidates[idx] for idx in query_idx]
        queries_db = ase.db.connect("queried_images.db")
        write_to_db(queries_db, queried_images)
        self.parent_calls += len(queried_images)
        return queried_images

    def sub_query_func(self, candidates_list):
        """
        Default random query strategy. Returns list of idx ints for query_func method from a
        given (possibly restricted) list of candidates.

        Designed to be overwritable
        """
        random.seed(self.query_seeds[self.iterations - 1])
        query_idx = random.sample(
            range(len(candidates_list)),
            self.samples_to_retrain,
        )
        return query_idx

    def check_terminate(self):
        if self.iterations >= self.max_iterations:
            return True
        else:
            if self.iterations > 0 and self.final_point_force <= self.max_evA:
                return True
        return False

    def get_uncertainty_tol(self):
        """
        Computes uncertainty tolerance to be used as the criteria for restriction

        Designed to be overwritable
        """
        return self.uncertainty_tol


class DynamicUncertaintyOffAL(UncertaintyOffAL):
    """
    Extends UncertaintyOffAL.
    Changes it to estimate the uncertainty threshold dynamically to restrict when to query
    Specifically, it will randomly query from within a dynamic uncertainty threshold based on a given tolerance

    Can serve as a parent class for other uncertainty offline learners
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

    ensemble: int
        The number of models in ensemble

    unertainty_tol: float
        dynamic threshold for uncertainty tolerance
    """

    def __str__(self):
        return "Dynamic-Uncertainty-Based Offline Learner"

    def get_uncertainty_tol(self):
        """
        Computes uncertainty tolerance to be used as the criteria for restriction

        Overwritten to compute uncertainty dynamically
        """
        base_uncertainty = np.nanmax(np.abs(self.training_data[0][0].get_forces()))
        for image in self.training_data:
            temp_uncertainty = np.nanmax(np.abs(image[0].get_forces()))
            if temp_uncertainty < base_uncertainty:
                base_uncertainty = temp_uncertainty
        uncertainty_tol = base_uncertainty * self.uncertainty_tol
        return uncertainty_tol
