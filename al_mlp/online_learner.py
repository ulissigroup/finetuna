import copy
import numpy as np
from ase.db import connect
from ase.calculators.singlepoint import SinglePointCalculator as sp
from ase.calculators.calculator import Calculator
from al_mlp.utils import convert_to_singlepoint, compute_with_calc
from al_mlp.bootstrap import non_bootstrap_ensemble
from al_mlp.ensemble_calc import EnsembleCalc
from al_mlp.calcs import DeltaCalc
from al_mlp.utils import copy_images

__author__ = "Muhammed Shuaibi"
__email__ = "mshuaibi@andrew.cmu.edu"


class OnlineActiveLearner(Calculator):
    """Online Active Learner
    Parameters
    ----------
     learner_params: dict
         Dictionary of learner parameters and settings.

     trainer: object
         An isntance of a trainer that has a train and predict method.

     parent_dataset: list
         A list of ase.Atoms objects that have attached calculators.
         Used as the first set of training data.

     parent_calc: ase Calculator object
         Calculator used for querying training data.

     n_ensembles: int.
          n_ensemble of models to make predictions.

     n_cores: int.
          n_cores used to train ensembles.

     parent_calc: ase Calculator object
         Calculator used for querying training data.

     base_calc: ase Calculator object
         Calculator used to calculate delta data for training.

     trainer_calc: uninitialized ase Calculator object
         The trainer_calc should produce an ase Calculator instance
         capable of force and energy calculations via TrainerCalc(trainer)
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        learner_params,
        trainer,
        parent_dataset,
        parent_calc,
        base_calc,
        n_ensembles=10,
        n_cores="max",
    ):
        Calculator.__init__(self)

        self.n_ensembles = n_ensembles
        self.parent_calc = parent_calc
        self.base_calc = base_calc
        self.calcs = [parent_calc, base_calc]
        self.trainer = trainer
        self.learner_params = learner_params
        self.n_cores = n_cores
        self.ensemble_sets, self.parent_dataset = non_bootstrap_ensemble(
            parent_dataset, n_ensembles=n_ensembles
        )
        self.init_training_data()
        self.ensemble_calc = EnsembleCalc.make_ensemble(
            self.ensemble_sets, self.trainer, self.base_calc, self.refs
        )

        self.uncertain_tol = learner_params["uncertain_tol"]
        self.parent_calls = 0
        self.init_training_data()

    def init_training_data(self):
        """
        Prepare the training data by attaching delta values for training.
        """
        raw_data = self.parent_dataset
        sp_raw_data = convert_to_singlepoint(raw_data)
        parent_ref_image = sp_raw_data[0]
        base_ref_image = compute_with_calc(sp_raw_data[:1], self.base_calc)[0]
        self.refs = [parent_ref_image, base_ref_image]
        self.delta_sub_calc = DeltaCalc(self.calcs, "sub", self.refs)
        self.ensemble_sets, self.parent_dataset = non_bootstrap_ensemble(
            compute_with_calc(sp_raw_data, self.delta_sub_calc),
            n_ensembles=self.n_ensembles,
        )

    def calculate(self, atoms, properties, system_changes):

        if len(self.parent_dataset) == 1 and np.all(
            self.parent_dataset[0].positions == atoms.positions
        ):
            # We only have one training data, and we are calculating the energy/force for that point
            self.results["energy"] = self.parent_dataset[0].get_potential_energy(
                apply_constraint=False
            )
            self.results["forces"] = self.parent_dataset[0].get_forces(
                apply_constraint=False
            )
            return

        Calculator.calculate(self, atoms, properties, system_changes)

        ensemble_calc_copy = copy.deepcopy(self.ensemble_calc)
        energy_pred = ensemble_calc_copy.get_potential_energy(atoms)
        force_pred = ensemble_calc_copy.get_forces(atoms)
        uncertainty = atoms.info["uncertainty"][0]
        uncertainty_tol = self.uncertain_tol
        if (
            "relative_variance" in self.learner_params
            and self.learner_params["relative_variance"]
        ):
            ensemble_calc_copy = copy.deepcopy(self.ensemble_calc)
            copied_images = copy_images(self.parent_dataset)
            base_uncertainty = 0
            for image in copied_images:
                ensemble_calc_copy.reset()
                ensemble_calc_copy.get_forces(image)
                if image.info["uncertainty"][0] > base_uncertainty:
                    base_uncertainty = image.info["uncertainty"][0]
            uncertainty_tol = self.uncertain_tol * base_uncertainty

        db = connect("dft_calls.db")

        print(
            "uncertainty: "
            + str(uncertainty)
            + ", uncertainty_tol: "
            + str(uncertainty_tol)
        )  # FIXME remove me
        if uncertainty >= uncertainty_tol or len(self.parent_dataset) == 1:
            print("DFT required")
            new_data = atoms.copy()
            new_data.set_calculator(copy.copy(self.parent_calc))
            # os.makedirs("./temp", exist_ok=True)
            # os.chdir("./temp")

            energy_pred = new_data.get_potential_energy(apply_constraint=False)
            force_pred = new_data.get_forces(apply_constraint=False)
            new_data.set_calculator(
                sp(atoms=new_data, energy=energy_pred, forces=force_pred)
            )
            # os.chdir(cwd)
            # os.system("rm -rf ./temp")

            try:
                db.write(new_data)
            except Exception:
                print("failed to write to db file")
                pass
            self.ensemble_sets, self.parent_dataset = non_bootstrap_ensemble(
                self.parent_dataset,
                compute_with_calc([new_data], self.delta_sub_calc),
                n_ensembles=self.n_ensembles,
            )

            self.ensemble_calc = EnsembleCalc.make_ensemble(
                self.ensemble_sets, self.trainer, self.base_calc, self.refs
            )

            self.parent_calls += 1
        else:
            print("")
            # db.write(None)
        self.results["energy"] = energy_pred
        self.results["forces"] = force_pred
