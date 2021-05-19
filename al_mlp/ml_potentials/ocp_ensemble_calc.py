import numpy as np
from ase.calculators.calculator import Calculator
import random
from al_mlp.ml_potentials.bootstrap import non_bootstrap_ensemble
import torch
import uuid

from ocpmodels.trainers.amp_xfer_trainer import OCPXTrainer

torch.multiprocessing.set_sharing_strategy("file_system")

__author__ = "Joe Musielewicz"
__email__ = "jmusiele@andrew.cmu.edu"


class OCPEnsembleCalc(Calculator):
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
        Calculator.__init__(self)
        self.amptorch_trainer = amptorch_trainer
        self.n_ensembles = n_ensembles

    def calculate_stats(self, energies, forces):
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
        Calculator.calculate(self, atoms, properties, system_changes)
        energies = []
        forces = []
        for predictor in self.trained_trainers:
            prediction = predictor.predict(atoms)
            energies.append(prediction["energy"].data.numpy()[0])
            forces.append(prediction["forces"].data.numpy())

        energies = np.array(energies)
        forces = np.array(forces)
        energy_pred, force_pred, max_forces_var, energy_var = self.calculate_stats(
            energies, forces
        )

        self.results["energy"] = energy_pred
        self.results["forces"] = force_pred
        atoms.info["energy_stds"] = energy_var ** 0.2
        atoms.info["max_force_stds"] = max_forces_var ** 0.5

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
            seed = args_list[2]
            uniqueid = args_list[3]

            trainer.model = OCPXTrainer.get_pretrained(
                training_dataset, seed, uniqueid, trainer.a2g_train
            )

            trainer.train(raw_data=training_dataset)
            # check_path = trainer.cp_dir
            # trainer = AtomsTrainer()
            # trainer.load_pretrained(checkpoint_path=check_path)
            # trainer_calc = trainer.get_calc()
            # return trainer_calc
            return trainer

        # split ensemble sets into separate args_lists, clone: trainer,
        # base calc and add to args_lists, add: refs to args_lists
        args_lists = []
        random.seed(self.amptorch_trainer.config["cmd"]["seed"])
        randomlist = [random.randint(0, 4294967295) for set in ensemble_sets]
        for i in range(len(ensemble_sets)):
            ensemble_set = ensemble_sets[i]
            random.seed(randomlist[i])
            random.shuffle(ensemble_set)

            trainer_copy = self.amptorch_trainer.copy()
            trainer_copy.config["cmd"]["seed"] = randomlist[i]
            trainer_copy.config["cmd"]["identifier"] = trainer_copy.config["cmd"][
                "identifier"
            ] + str(uuid.uuid4())

            args_lists.append(
                (
                    ensemble_set,
                    trainer_copy,
                    randomlist[i],
                    trainer_copy.model.config["cmd"]["identifier"] + str(uuid.uuid4()),
                )
            )

        # map training method, returns array of delta calcs
        trained_trainers = []
        if self.executor is not None:
            futures = []
            for args_list in args_lists:
                big_future = self.executor.scatter(args_list)
                futures.append(self.executor.submit(train_and_combine, big_future))
            trained_trainers = [future.result() for future in futures]
        else:
            for args_list in args_lists:
                trained_trainers.append(train_and_combine(args_list))

        # call init to construct ensemble calc from array of delta calcs
        self.trained_trainers = trained_trainers

    @classmethod
    def set_executor(cls, executor):
        cls.executor = executor
