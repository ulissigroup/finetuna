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
        max_forces_var = np.max(np.var(forces, axis=0))
        max_energy_var = np.var(energies)
        return energy_median, forces_median, max_forces_var #change back to max forces var

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        energies = []
        forces = []

        def evaluate_ef(atoms_image):
            return (calc.get_potential_energy(atoms_image), calc.get_forces(atoms))
        if self.executor is not None:
            futures = []
            for calc in self.trained_calcs:
                futures.append(self.executor.submit(evaluate_ef, atoms))
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
    def make_ensemble(cls, ensemble_sets, trainer, base_calc, refs):
        """
        Uses Dask to parallelize, must have previously set up cluster, image to use, and pool of workers
        """
        #method for training trainer on ensemble sets, then create neural net calc, combine with base calc, return additive delta calc
        def train_and_combine(args_tuple):
            ensemble_set = args_tuple[0]
            trainer = args_tuple[1]
            base_calc = args_tuple[2]
            refs = args_tuple[3]

            trainer.train(raw_data=ensemble_set)
            check_path = trainer.cp_dir
            trainer = AtomsTrainer()
            trainer.load_pretrained(checkpoint_path=check_path)
            trained_calc = DeltaCalc((trainer.get_calc(),base_calc),"add",refs)
            return trained_calc

        #split ensemble sets into separate tuples, clone: trainer, base calc and add to tuples, add: refs to tuples
        tuples = []
        random.seed(trainer.config["cmd"]["seed"])
        randomlist = [random.randint(0,4294967295) for set in ensemble_sets]
        for i in range(len(ensemble_sets)):
            set = ensemble_sets[i]
            trainer_copy = AtomsTrainer(copy.deepcopy(trainer.config))
            trainer_copy.config["cmd"]["seed"] = randomlist[i]
            trainer_copy.load_rng_seed()
            base_calc_copy = copy.deepcopy(base_calc)
            refs_copy = copy_images(refs) 
            tuples.append((set, trainer_copy, base_calc_copy, refs_copy))
        
        #map training method, returns array of delta calcs
        # tuples_bag = daskbag.from_sequence(tuples)
        # tuples_bag_computed = tuples_bag.map(train_and_combine)
        # if "single-threaded" in  trainer.config["cmd"] and trainer.config["cmd"]["single-threaded"]:
        #     trained_calcs = tuples_bag_computed.compute(scheduler='single-threaded')
        # else:
        #     trained_calcs = tuples_bag_computed.compute()
        trained_calcs = []
        if cls.executor is not None:
            futures = []
            for tuple in tuples:
                futures.append(cls.executor.submit(train_and_combine, tuple))
            trained_calcs = [future.result() for future in futures]
        else:
            for tuple in tuples:
                trained_calcs.append(train_and_combine(tuple))

        #call init to construct ensemble calc from array of delta calcs
        return cls(trained_calcs)

    @classmethod
    def set_executor(cls,executor):
        cls.executor = executor
    