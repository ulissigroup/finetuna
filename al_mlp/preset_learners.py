import ase
import numpy as np
from al_mlp.calcs import DeltaCalc
from al_mlp.utils import write_to_db
from al_mlp.bootstrap import bootstrap_ensemble
from al_mlp.ensemble_calc import EnsembleCalc
from al_mlp.offline_active_learner import OfflineActiveLearner


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
        self, learner_params, trainer, training_data, parent_calc, base_calc, ensemble
    ):
        super().__init__(learner_params, trainer, training_data, parent_calc, base_calc)

        self.ensemble = ensemble
        assert isinstance(ensemble, int) and ensemble > 1, "Invalid ensemble!"
        self.training_data, self.parent_dataset = bootstrap_ensemble(
            self.training_data, n_ensembles=ensemble
        )
        self.parent_calls = 0

    def do_before_train(self):
        if self.iterations > 0:
            queried_images = self.query_func()
            self.parent_dataset, self.training_data = self.add_data(queried_images)
            self.parent_calls += len(queried_images)
        self.fn_label = f"{self.file_dir}{self.filename}_iter_{self.iterations}"
        self.ensemble_sets = self.training_data

    def do_train(self):
        self.make_ensemble()

    def do_after_train(self):
        self.atomistic_method.run(calc=self.trained_calc, filename=self.fn_label)
        self.sample_candidates = list(
            self.atomistic_method.get_trajectory(filename=self.fn_label)
        )

        self.terminate = self.check_terminate()
        self.iterations += 1

    def do_after_learn(self):
        pass

    def query_func(self):
        queries_db = ase.db.connect("queried_images.db")
        uncertainty = np.array(
            [atoms.info["uncertainty"][0] for atoms in self.sample_candidates]
        )
        query_idx = np.argpartition(uncertainty, -1 * self.samples_to_retrain)[
            -1 * self.samples_to_retrain :
        ]
        queried_images = [self.sample_candidates[idx] for idx in query_idx]
        # write_to_db(queries_db, queried_images) bugged unique ID
        return queried_images

    def add_data(self, queried_images):
        for query in queried_images:
            # bug here maybe?: self.training_data and self.parent_dataset
            # are being overwritten in every iteration.
            self.training_data, self.parent_dataset = bootstrap_ensemble(
                self.parent_dataset,
                self.training_data,
                query,
                n_ensembles=self.ensemble,
            )
        return self.parent_dataset, self.training_data

    def make_ensemble(self):
        trained_calcs = []
        for dataset in self.ensemble_sets:
            self.trainer.train(dataset)
            trainer_calc = self.make_trainer_calc()
            trained_calcs.append(
                DeltaCalc([trainer_calc, self.base_calc], "add", self.refs)
            )
        ensemble_calc = EnsembleCalc(trained_calcs, self.trainer)
        self.trained_calc = ensemble_calc
