from al_mlp.preset_learners.ensemble_learner import EnsembleLearner
import ase
import random
from al_mlp.utils import write_to_db
import numpy as np


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
        ensemble=5,
        uncertainty_tol=0.05,
    ):
        super().__init__(
            learner_params,
            trainer,
            training_data,
            parent_calc,
            base_calc,
            ensemble,
        )
        self.uncertainty_tol = uncertainty_tol

    def __str__(self):
        return "Default (Static) Uncertainty-Based Offline Learner"

    def query_func(self):
        """
        Boiler-plate code for querying
        Offloads two aspects of the undertainty based querying strategy to two sub-functions
            restrict_candidates()
                temporarily restrict acceptable querying candidates based on uncertainty_tol
            sub_query_func()
                generate list of idx of points to query from the acceptable querying list
        """
        restricted_candidates = self.restrict_candidates()
        query_idx = self.sub_query_func(restricted_candidates)
        queried_images = [restricted_candidates[idx] for idx in query_idx]
        queries_db = ase.db.connect("queried_images.db")
        write_to_db(queries_db, queried_images)
        self.parent_calls += len(queried_images)
        return queried_images

    def restrict_candidates(self):
        """
        Boiler-plate code for getting a restricted candidate set
        Restricts candidates based on their uncertainty being below the given tolerance
            Offloads computation of this to tolerance to the function get_uncertainty_tol()
        By default adds to the restricted set the lowest uncertainty candidates if set is not large enough
        """
        restricted_candidates = []
        remaining_candidates = []
        for i in range(len(self.sample_candidates)):
            uncertainty = self.sample_candidates[i].info["uncertainty"][0] ** 0.5
            if uncertainty < self.get_uncertainty_tol():
                restricted_candidates.append(self.sample_candidates[i])
            else:
                remaining_candidates.append(self.sample_candidates[i])
        # if there aren't enough candidates based on criteria, get the lowest uncertainty remaining
        # candidates to return
        if len(restricted_candidates) < self.samples_to_retrain:
            print("Warning: not enough sample candidates which meet criteria")
            remaining_candidates.sort(
                key=lambda candidate: candidate.info["uncertainty"][0]
            )
            restricted_candidates.extend(
                remaining_candidates[
                    : self.samples_to_retrain - len(restricted_candidates)
                ]
            )
        return restricted_candidates

    def get_uncertainty_tol(self):
        """
        Computes uncertainty tolerance to be used as the criteria for restriction

        Designed to be overwritable
        """
        return self.uncertainty_tol

    def sub_query_func(self, candidates_list):
        """
        Default random query strategy. Returns list of idx ints for query_func method from a
        given (possibly restricted) list of candidates.

        Designed to be overwritable
        """
        random.seed()
        query_idx = random.sample(
            range(len(candidates_list)),
            self.samples_to_retrain,
        )
        return query_idx


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
        base_uncertainty = self.training_data[0]
        for image in self.training_data:
            temp_uncertainty = np.nanmax(np.abs(image.get_forces()))
            if temp_uncertainty < base_uncertainty:
                base_uncertainty = temp_uncertainty
        uncertainty_tol = base_uncertainty * self.uncertainty_tol
        return uncertainty_tol
