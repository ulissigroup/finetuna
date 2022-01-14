import random
from ase.io.trajectory import Trajectory
from al_mlp.offline_learner.offline_learner import OfflineLearner
from ase.optimize.bfgs import BFGS
from al_mlp.atomistic_methods import Relaxation
from al_mlp.base_calcs.dummy import Dummy
from al_mlp.calcs import DeltaCalc
from al_mlp.utils import compute_with_calc
import numpy as np
from ase.calculators.calculator import Calculator
from al_mlp.logger import Logger


class OfflineDeltaLearner(OfflineLearner):
    """Offline Delta Learner.
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
        self.base_calc = base_calc
        self.calcs = [parent_calc, base_calc]

        OfflineLearner.__init__(
            self,
            learner_params,
            training_data,
            ml_potential,
            parent_calc,
            mongo_db=mongo_db,
            optional_config=optional_config,
        )

    def init_logger(self, mongo_db, optional_config):
        if mongo_db is None:
            mongo_db = {"offline_learner": None}
        self.logger = Logger(
            learner_params=self.learner_params,
            ml_potential=self.ml_potential,
            parent_calc=self.parent_calc,
            base_calc=self.base_calc,
            mongo_db_collection=mongo_db["offline_learner"],
            optional_config=optional_config,
        )

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
        for image in self.initial_data:
            raw_data.append(image)

        # run a trajectory with no training data: just the base model to sample from
        self.training_data = []
        self.fn_label = f"{self.file_dir}{self.filename}_iter_{self.iterations}"
        self.do_after_train()

        # add initial data to training dataset
        self.add_data(raw_data, None)
        self.initial_image_energy = self.refs[0].get_potential_energy()

    def add_data(self, queried_images, query_idx):
        """
        Attaches calculators to the queried images and runs calculate
        Adds the training-ready images to the training dataset
        Returns the  images in a list called new_dataset
        """
        new_dataset = compute_with_calc(queried_images, self.delta_sub_calc)
        self.training_data += new_dataset
        self.parent_calls += len(new_dataset)

        un_delta_new_dataset = []
        for image in new_dataset:
            add_delta_calc = DeltaCalc([image.calc, self.base_calc], "add", self.refs)
            [un_delta_image] = compute_with_calc([image], add_delta_calc)
            un_delta_new_dataset.append(un_delta_image)

        self.log_parent_data(un_delta_new_dataset, query_idx)

        return un_delta_new_dataset

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

        delta_calc = DeltaCalc([calc, self.base_calc], "add", self.refs)
        return delta_calc
