import numpy as np
from ase.calculators.calculator import Calculator
import random
import copy
from amptorch.trainer import AtomsTrainer
import torch
import uuid

torch.multiprocessing.set_sharing_strategy("file_system")

__author__ = "Muhammed Shuaibi"
__email__ = "mshuaibi@andrew.cmu.edu"


class EnsembleCalc(Calculator):
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

    implemented_properties = ["energy", "forces", "uncertainty"]
    executor = None

    def __init__(self, trained_calcs):
        Calculator.__init__(self)
        self.trained_calcs = trained_calcs

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
        # max_energy_var = np.nanvar(energies)
        return (
            energy_median,
            forces_median,
            max_forces_var,
        )  # change back to max forces var

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        energies = []
        forces = []

        def evaluate_ef(args_list):
            """
            accepts a args_list of an atoms image and a calculator
            evaluates the energies and forces of the atom using the calculator
            and returns them as a args_list
            """
            atoms_image = args_list[0]
            calc = args_list[1]
            calc.trainer.config["dataset"]["save_fps"] = False
            return (
                calc.get_potential_energy(atoms_image),
                calc.get_forces(atoms_image),
            )

        # Forward pass of the ensemble is so fast that passing it in and out of futures is slower
        if False:  # self.executor is not None:
            futures = []
            for calc in self.trained_calcs:
                big_future = self.executor.scatter((atoms, calc))
                futures.append(self.executor.submit(evaluate_ef, big_future))
            energies = [future.result()[0] for future in futures]
            forces = [future.result()[1] for future in futures]
        else:
            for calc in self.trained_calcs:
                energies.append(calc.get_potential_energy(atoms))
                forces.append(calc.get_forces(atoms))

        energies = np.array(energies)
        forces = np.array(forces)
        energy_pred, force_pred, uncertainty = self.calculate_stats(energies, forces)

        self.results["energy"] = energy_pred
        self.results["forces"] = force_pred
        atoms.info["uncertainty"] = np.array([uncertainty])

    @classmethod
    def make_ensemble(cls, ensemble_sets, trainer):
        """
        Uses Dask to parallelize, must have previously set up cluster,
        image to use, and pool of workers
        """

        def train_and_combine(args_list):
            """
            method for training trainer on ensemble sets, then create neural net calc,
            returns trained calc
            """
            ensemble_set = args_list[0]
            trainer = args_list[1]

            trainer.train(raw_data=ensemble_set)
            check_path = trainer.cp_dir
            trainer = AtomsTrainer()
            trainer.load_pretrained(checkpoint_path=check_path)
            trainer_calc = trainer.get_calc()
            return trainer_calc

        # split ensemble sets into separate args_lists, clone: trainer,
        # base calc and add to args_lists, add: refs to args_lists
        args_lists = []
        random.seed(trainer.config["cmd"]["seed"])
        randomlist = [random.randint(0, 4294967295) for set in ensemble_sets]
        for i in range(len(ensemble_sets)):
            set = ensemble_sets[i]

            copy_config = copy.deepcopy(trainer.config)
            copy_config["cmd"]["seed"] = randomlist[i]
            copy_config["cmd"]["identifier"] = copy_config["cmd"]["identifier"] + str(
                uuid.uuid4()
            )

            trainer_copy = AtomsTrainer(copy_config)
            args_lists.append((set, trainer_copy))

        # map training method, returns array of delta calcs
        trained_calcs = []
        if cls.executor is not None:
            futures = []
            for args_list in args_lists:
                big_future = cls.executor.scatter(args_list)
                futures.append(cls.executor.submit(train_and_combine, big_future))
            trained_calcs = [future.result() for future in futures]
        else:
            for args_list in args_lists:
                trained_calcs.append(train_and_combine(args_list))

        # call init to construct ensemble calc from array of delta calcs
        return cls(trained_calcs)

    @classmethod
    def set_executor(cls, executor):
        cls.executor = executor
