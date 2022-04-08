from finetuna.online_learner.online_learner import OnlineLearner
from finetuna.calcs import DeltaCalc
from finetuna.utils import convert_to_singlepoint, subtract_deltas
from finetuna.logger import Logger
from finetuna.utils import compute_with_calc
from copy import deepcopy

__author__ = "Joseph Musielewicz"
__email__ = "al.mlp.package@gmail.com"


class DeltaLearner(OnlineLearner):
    def __init__(
        self,
        learner_params,
        parent_dataset,
        ml_potential,
        parent_calc,
        base_calc,
        mongo_db=None,
        optional_config=None,
    ):
        self.base_calc = base_calc
        self.refs = None

        OnlineLearner.__init__(
            self,
            learner_params,
            parent_dataset,
            ml_potential,
            parent_calc,
            mongo_db=mongo_db,
            optional_config=optional_config,
        )

    def init_logger(self, mongo_db, optional_config):
        self.logger = Logger(
            learner_params=self.learner_params,
            ml_potential=self.ml_potential,
            parent_calc=self.parent_calc,
            base_calc=self.base_calc,
            mongo_db_collection=mongo_db["online_learner"],
            optional_config=optional_config,
        )

    def init_refs(self, initial_structure):
        self.parent_ref = initial_structure.copy()
        self.parent_ref.calc = deepcopy(initial_structure.calc)

        self.base_ref = compute_with_calc([initial_structure.copy()], self.base_calc)[0]

        self.refs = [self.parent_ref, self.base_ref]

        self.add_delta_calc = DeltaCalc(
            [self.ml_potential, self.base_calc],
            "add",
            self.refs,
        )

    def get_ml_calc(self):
        self.ml_potential.reset()
        self.add_delta_calc.reset()
        return self.add_delta_calc

    def get_ml_prediction(self, atoms):
        """
        Helper function which takes an atoms object with no calc attached.
        Makes an Ml prediction.
        Performs a delta add operation since the ML model was trained on delta sub data.
        Returns it with a delta ML potential predicted singlepoint.
        Designed to be overwritten by DeltaLearner which needs to modify ML predictions.
        """
        atoms_copy = atoms.copy()
        atoms_copy.set_calculator(self.ml_potential)
        (atoms_with_info,) = convert_to_singlepoint([atoms_copy])
        atoms_copy.set_calculator(self.add_delta_calc)
        (atoms_ML,) = convert_to_singlepoint([atoms_copy])
        for key, value in atoms_with_info.info.items():
            atoms_ML.info[key] = value
        return atoms_ML

    def add_to_dataset(self, new_data):
        """
        Helper function which takes an atoms object with parent singlepoint attached.
        Performs a delta sub operation on the parent data so that the ML model will train on delta sub data.
        And adds new parent data to the training set.
        Returns the partial dataset just added.
        Designed to overwritten by DeltaLearner which needs to modify data added to training set.
        """
        if self.refs is None:
            self.init_refs(new_data)

        (delta_sub_data,) = subtract_deltas([new_data], self.base_calc, self.refs)
        partial_dataset = [delta_sub_data]
        self.parent_dataset += partial_dataset
        return partial_dataset
