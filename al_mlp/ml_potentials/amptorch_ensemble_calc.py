import numpy as np
import random
import copy
from finetuna.ml_potentials.ml_potential_calc import MLPCalc
from amptorch.trainer import AtomsTrainer
from finetuna.ml_potentials.bootstrap import non_bootstrap_ensemble
import torch
import uuid

torch.multiprocessing.set_sharing_strategy("file_system")

__author__ = "Muhammed Shuaibi"
__email__ = "mshuaibi@andrew.cmu.edu"


class AmptorchEnsembleCalc(MLPCalc):
    """Atomistics Machine-Learning Potential (AMP) ASE calculator
    Parameters
    ----------
     model : object
         Class representing the regression model. Input arguments include training
         images, descriptor type, and force_coefficient. Model structure and training schemes can be
         modified directly within the class.

     label : str
         Location to save the trained model.

    """

    implemented_properties = ["energy", "forces", "max_force_stds", "energy_stds"]
    executor = None

    def __init__(self, amptorch_trainer, n_ensembles):
        MLPCalc.__init__(self, mlp_params=amptorch_trainer.config)
        self.amptorch_trainer = amptorch_trainer
        self.n_ensembles = n_ensembles

    def calculate_stats(self, energies, forces):
        # energies_mean = np.mean(energies,axis=0)
        # forces_mean = np.mean(forces,axis=0)
        # abs_max_force_idx = np.argmax(np.abs(forces_mean))
        # direction_idx = int(abs_max_force_idx%3)
        # force_idx = int((abs_max_force_idx-direction_idx)/3)
        # abs_max_force = forces_mean[force_idx][direction_idx]

        median_idx = np.argsort(energies)[len(energies) // 2]
        energy_median = energies[median_idx]
        forces_median = forces[median_idx]
        max_forces_var = np.nanmax(np.nanvar(forces, axis=0))
        energy_var = np.nanvar(energies)
        return (
            energy_median,
            forces_median,
            max_forces_var,
            energy_var,
        )

    def calculate(self, atoms, properties, system_changes):
        MLPCalc.calculate(
            self, atoms=atoms, properties=properties, system_changes=system_changes
        )
        energies = []
        forces = []
        for calc in self.trained_calcs:
            energies.append(calc.get_potential_energy(atoms))
            forces.append(calc.get_forces(atoms))

        energies = np.array(energies)
        forces = np.array(forces)
        energy_pred, force_pred, max_forces_var, energy_var = self.calculate_stats(
            energies, forces
        )

        self.results["energy"] = energy_pred
        self.results["forces"] = force_pred
        atoms.info["energy_stds"] = energy_var**0.2
        atoms.info["max_force_stds"] = max_forces_var**0.5

        # self.results["max_force_stds"] = np.array(max_forces_var) ** 0.5
        # self.results["energy_stds"] = energy_var ** 0.2

    def train(self, parent_dataset, new_dataset=None):
        """
        Uses Dask to parallelize, must have previously set up cluster,
        image to use, and pool of workers
        """

        ensemble_sets, parent_dataset = non_bootstrap_ensemble(
            parent_dataset, n_ensembles=self.n_ensembles
        )

        def train_and_combine(args_list):
            """
            method for training trainer on ensemble sets, then create neural net calc,
            returns trained calc
            """
            training_dataset = args_list[0]
            trainer = args_list[1]

            trainer.train(raw_data=training_dataset)
            check_path = trainer.cp_dir
            trainer = AtomsTrainer()
            trainer.load_pretrained(checkpoint_path=check_path)
            trainer_calc = trainer.get_calc()
            return trainer_calc

        # split ensemble sets into separate args_lists, clone: trainer,
        # base calc and add to args_lists, add: refs to args_lists
        args_lists = []
        random.seed(self.amptorch_trainer.config["cmd"]["seed"])
        randomlist = [random.randint(0, 4294967295) for set in ensemble_sets]
        for i in range(len(ensemble_sets)):
            ensemble_set = ensemble_sets[i]

            copy_config = copy.deepcopy(self.amptorch_trainer.config)
            copy_config["cmd"]["seed"] = randomlist[i]
            copy_config["cmd"]["identifier"] = copy_config["cmd"]["identifier"] + str(
                uuid.uuid4()
            )

            trainer_copy = AtomsTrainer(copy_config)
            args_lists.append((ensemble_set, trainer_copy))

        # map training method, returns array of delta calcs
        trained_calcs = []
        if self.executor is not None:
            futures = []
            for args_list in args_lists:
                big_future = self.executor.scatter(args_list)
                futures.append(self.executor.submit(train_and_combine, big_future))
            trained_calcs = [future.result() for future in futures]
        else:
            for args_list in args_lists:
                trained_calcs.append(train_and_combine(args_list))

        # call init to construct ensemble calc from array of delta calcs
        self.trained_calcs = trained_calcs

    @classmethod
    def set_executor(cls, executor):
        cls.executor = executor
