import numpy as np
from ase.calculators.calculator import Calculator
import random
from al_mlp.calcs import DeltaCalc
import copy
import dask.bag as daskbag
from al_mlp.utils import copy_images
from amptorch.trainer import AtomsTrainer
from torch.multiprocessing import Pool
import torch
torch.multiprocessing.set_sharing_strategy("file_system")
from dask.distributed import Client
from concurrent.futures import ThreadPoolExecutor

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

    def __init__(self, trained_calcs):
        Calculator.__init__(self)
        self.trained_calcs = trained_calcs

    def calculate_stats(self, energies, forces):
        median_idx = np.argsort(energies)[len(energies) // 2]
        energy_median = energies[median_idx]
        forces_median = forces[median_idx]
        max_forces_var = np.max(np.var(forces, axis=0))
        max_energy_var = np.var(energies)
        return energy_median, forces_median, max_forces_var #change back to max forces var

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        energies = []
        forces = []

        for calc in self.trained_calcs:
            energies.append(calc.get_potential_energy(atoms))
            forces.append(calc.get_forces(atoms))

        energies = np.array(energies)
        forces = np.array(forces)
        energy_pred, force_pred, uncertainty = self.calculate_stats(energies, forces)

        self.results["energy"] = energy_pred
        self.results["forces"] = force_pred
        atoms.info["uncertainty"] = np.array([uncertainty])


def train_and_combine(args_tuple):
        ensemble_set = args_tuple[0]
        trainer = args_tuple[1]

        trainer.train(raw_data=ensemble_set)
        check_path = trainer.cp_dir
        trainer = AtomsTrainer()
        trainer.load_pretrained(checkpoint_path=check_path)
        return trainer


def make_ensemble(ensemble_sets, trainer):
    parallel_args = []
    random.seed(trainer.config["cmd"]["seed"])
    randomlist = [random.randint(0,4294967295) for set in ensemble_sets]
    for i in range(len(ensemble_sets)):
        ensemble_train_data = ensemble_sets[i]
        config = copy.deepcopy(trainer.config)
        config["cmd"]["seed"] = randomlist[i]
        trainer_copy = AtomsTrainer(config)
        parallel_args.append((ensemble_train_data, trainer_copy))

    trained_calcs = []
    for args in parallel_args:
        trained = train_and_combine(args)
        trained_calcs.append(trained.get_calc())

    return trained_calcs
