import numpy as np
import copy
import torch
from al_mlp.utils import (
    compute_with_calc,
    write_to_db,
    subtract_deltas,
    convert_to_singlepoint,
)

from al_mlp.calcs import DeltaCalc
import random
import ase


# from al_mlp.utils import write_to_db
from al_mlp.ml_potentials.bootstrap import bootstrap_ensemble
from al_mlp.ml_potentials.bootstrap import non_bootstrap_ensemble

from al_mlp.ml_potentials.amptorch_ensemble_calc import AmptorchEnsembleCalc
from al_mlp.offline_learner.offline_learner import OfflineActiveLearner

# from torch.multiprocessing import Pool

torch.multiprocessing.set_sharing_strategy("file_system")


class EnsembleLearner(OfflineActiveLearner):
    """Offline Active Learner using ensemble and to query
    data with the most uncertainty.
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

    ensemble: int
        The number of models in ensemble
    """

    def __init__(
        self,
        learner_params,
        trainer,
        training_data,
        parent_calc,
        base_calc,
        ml_potential,
    ):
        super().__init__(learner_params, trainer, training_data, parent_calc, base_calc)

        # assert isinstance(ensemble, int) and ensemble > 1, "Invalid ensemble!"
        # self.ncores = self.learner_params.get("ncores", ensemble)
        self.ml_potential = ml_potential
        self.ensemble = self.ml_potential.n_ensembles
        self.parent_calls = 0
        for image in self.training_data:
            print("initial trainint data ", image.get_potential_energy())

    def do_before_train(self):
        if self.iterations > 0:
            queried_images = self.query_func()
            queried_images = compute_with_calc(queried_images, self.delta_sub_calc)
            self.training_data += queried_images
            for image in self.training_data:
                print("trainint data energy ", image.get_potential_energy())
            # self.parent_dataset, self.training_data = self.add_data(queried_images)
            self.parent_calls += len(queried_images)
        self.fn_label = f"{self.file_dir}{self.filename}_iter_{self.iterations}"
        # self.ensemble_sets = self.training_data

    def do_train(self):
        self.ml_potential.train(self.training_data)
        self.trained_calc = DeltaCalc(
            [self.ml_potential, self.base_calc], "add", self.refs
        )

    def check_final_force(self):
        pass

    def do_after_train(self):
        self.atomistic_method.run(calc=self.trained_calc, filename=self.fn_label)
        self.sample_candidates = list(
            self.atomistic_method.get_trajectory(filename=self.fn_label)
        )
        self.check_final_force()
        self.terminate = self.check_terminate()
        self.iterations += 1

    def do_after_learn(self):
        pass

    def query_func(self):
        if self.iterations > 1:
            uncertainty = np.array(
                [atoms.info["max_force_stds"] for atoms in self.sample_candidates]
            )
            n_retrain = self.samples_to_retrain
            query_idx = np.argpartition(uncertainty, -1 * n_retrain)[-n_retrain:]
            queried_images = [self.sample_candidates[idx] for idx in query_idx]
        else:
            query_idx = random.sample(
                range(1, len(self.sample_candidates)),
                self.samples_to_retrain,
            )
            queried_images = [self.sample_candidates[idx] for idx in query_idx]

        return queried_images

    # def add_data(self, queried_images):
    #     for query in queried_images:
    #         self.training_data, self.parent_dataset = non_bootstrap_ensemble(
    #             self.parent_dataset,
    #             query,
    #             n_ensembles=self.ensemble,
    #         )
    #     return self.parent_dataset, self.training_data

    # def ensemble_train_trainer(self, dataset):
    #     trainer = copy.deepcopy(self.trainer)
    #     trainer.train(dataset)
    #     trainer_calc = self.make_trainer_calc(trainer)
    #     trained_calc = DeltaCalc([trainer_calc, self.base_calc], "add", self.refs)
    #     return trained_calc

    # def make_ensemble(self):
    #     pool = Pool(self.ncores)
    #     trained_calcs = pool.map(self.ensemble_train_trainer,
    # self.ensemble_sets)
    #     pool.close()
    #     pool.join()
    #     ensemble_calc = EnsembleCalc(trained_calcs)
    #     self.trained_calc = ensemble_calc
    # def make_ensemble(self):
    #     trained_calcs = []
    #     for ensemble_set in self.ensemble_sets:
    #         trained_calcs.append(self.ensemble_train_trainer(ensemble_set))
    #     ensemble_calc= EnsembleCalc(trained_calcs)
    #     self.trained_calc= ensemble_calc
