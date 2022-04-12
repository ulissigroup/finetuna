import ase
from ase.calculators.emt import EMT
from ase.neb import SingleCalculatorNEB
from ase.optimize import BFGS
from ase.io import read
from ase.build import fcc100, add_adsorbate
from ase.constraints import FixAtoms
from ase.neb import NEBTools
import matplotlib.pyplot as plt
import numpy as np
import copy
import torch

from finetuna.offline_active_learner import OfflineActiveLearner
from finetuna.base_calcs.morse import MultiMorse

from amptorch.trainer import AtomsTrainer


class NEBcalc:
    def __init__(self, starting_images, ml2relax=True, intermediate_samples=3):
        """
        Computes a NEB given an initial and final image.

        Parameters
        ----------
        starting_images: list. Initial and final images to be used for the NEB.

        ml2relax: boolean. True to use ML to relax the initial and final structure guesses.
        False if initial and final structures were relaxed beforehand.

        intermediate_samples:
        int. Number of intermediate samples to be used in constructing the NEB"""

        self.starting_images = copy.deepcopy(starting_images)
        self.ml2relax = ml2relax
        self.intermediate_samples = intermediate_samples

    def run(self, calc, filename):
        """
        Runs NEB calculations.
        Parameters
        ----------
        calc: object. Calculator to be used to run method.
        filename: str. Label to save generated trajectory files."""

        initial = self.starting_images[0].copy()
        final = self.starting_images[-1].copy()
        if self.ml2relax:
            # Relax initial and final images
            ml_initial = initial
            ml_initial.set_calculator(calc)
            ml_final = final
            ml_final.set_calculator(calc)
            print("BUILDING INITIAL")
            qn = BFGS(
                ml_initial, trajectory="initial.traj", logfile="initial_relax_log.txt"
            )
            qn.run(fmax=0.01, steps=100)
            print("BUILDING FINAL")
            qn = BFGS(ml_final, trajectory="final.traj", logfile="final_relax_log.txt")
            qn.run(fmax=0.01, steps=100)
            initial = ml_initial.copy()
            final = ml_final.copy()

        initial.set_calculator(calc)
        final.set_calculator(calc)

        images = [initial]
        for i in range(self.intermediate_samples):
            image = initial.copy()
            image.set_calculator(calc)
            images.append(image)
        images.append(final)

        print("NEB BEING BUILT")
        neb = SingleCalculatorNEB(images)
        neb.interpolate()
        print("NEB BEING OPTIMISED")
        opti = BFGS(neb, trajectory=filename + ".traj", logfile="al_neb_log.txt")
        opti.run(fmax=0.01, steps=100)
        print("NEB DONE")

        """
      The following code is used to visualise the NEB at every iteration
      """

        built_neb = NEBTools(images)
        barrier, dE = built_neb.get_barrier()
        # max_force = built_neb.get_fmax()
        # fig = built_neb.plot_band()
        plt.show()

    def get_trajectory(self, filename):
        atom_list = []
        trajectory = ase.io.Trajectory(filename + ".traj")
        for atom in trajectory:
            atom_list.append(atom)
        return atom_list


# https://wiki.fysik.dtu.dk/ase/tutorials/neb/diffusion.html#diffusion-tutorial
# Surface Diffusion Energy Barriers
# Building your structure


def construct_geometries(parent_calc, ml2relax):
    counter_calc = parent_calc
    # Initial structure guess
    initial_slab = fcc100("Cu", size=(2, 2, 3))
    add_adsorbate(initial_slab, "C", 1.7, "hollow")
    initial_slab.center(axis=2, vacuum=4.0)
    mask = [atom.tag > 1 for atom in initial_slab]
    initial_slab.set_constraint(FixAtoms(mask=mask))

    initial_slab.set_calculator(counter_calc)

    # Final structure guess
    final_slab = initial_slab.copy()
    final_slab[-1].x += final_slab.get_cell()[0, 0] / 3
    final_slab.set_calculator(counter_calc)
    if not ml2relax:
        print("BUILDING INITIAL")
        qn = BFGS(
            initial_slab, trajectory="initial.traj", logfile="initial_relax_log.txt"
        )
        qn.run(fmax=0.01, steps=100)
        print("BUILDING FINAL")
        qn = BFGS(final_slab, trajectory="final.traj", logfile="final_relax_log.txt")
        qn.run(fmax=0.01, steps=100)
        initial_slab = read("initial.traj", "-1")
        final_slab = read("final.traj", "-1")
        # If there is already a pre-existing initial and
        # final relaxed parent state we can read that to use as a starting point
        # initial_slab = read("/content/parent_initial.traj")
        # final_slab = read("/content/parent_final.traj")
    else:
        initial_slab = initial_slab
        final_slab = final_slab

    # initial_force_calls = counter_calc.force_calls
    return initial_slab, final_slab  # , initial_force_calls


'''
class NEBLearner(OfflineActiveLearner):
    def __init__(self, learner_params, trainer, training_data, parent_calc, base_calc):
        super().__init__(learner_params, trainer, training_data, parent_calc, base_calc)
        self.parent_calls = 0
    def check_terminate(self):
        """
        Default termination function.
        """
        if self.iterations >= self.max_iterations:
            return True
        return False

    def query_func(self):
        """
        NEB query strategy.
        """
        queries_db = ase.db.connect("queried_images.db")
        query_idx = random.sample(range(1, len(self.sample_candidates)), self.samples_to_retrain)
        queried_images = [self.sample_candidates[idx] for idx in query_idx]
        #write_to_db(queries_db,queried_images)
        return queried_images
'''


torch.set_num_threads(1)

parent_calc = EMT()

Gs = {
    "default": {
        "G2": {
            "etas": np.logspace(np.log10(0.05), np.log10(5.0), num=4),
            "rs_s": [0],
        },
        "G4": {"etas": [0.005], "zetas": [1.0, 4.0], "gammas": [1.0, -1.0]},
        "cutoff": 6,
    },
}

ml2relax = True
# use machine learning to relax the initial and
# final states rather than DFT as is the norm
total_neb_images = 5
# N + 2 where N is the number of intermediate images and
# 2 is for initial and final structures
initial, final = construct_geometries(parent_calc=parent_calc, ml2relax=ml2relax)

images = [initial]
images.append(final)

elements = ["Cu", "C"]
config = {
    "model": {"get_forces": True, "num_layers": 3, "num_nodes": 20},
    "optim": {
        "device": "cpu",
        "force_coefficient": 0.04,
        "lr": 1e-2,
        "batch_size": 1000,
        "epochs": 200,
        "loss": "mse",
        "metric": "mse",
        "optimizer": torch.optim.LBFGS,
    },
    "dataset": {
        "raw_data": images,
        "val_split": 0,
        "elements": elements,
        "fp_params": Gs,
        "save_fps": True,
        "scaling": {"type": "standardize", "range": (0, 1)},
    },
    "cmd": {
        "debug": False,
        "run_dir": "./",
        "seed": 1,
        "identifier": "test",
        "verbose": True,
        "logger": False,
        "dtype": torch.DoubleTensor,
    },
}

trainer = AtomsTrainer(config)

# building base morse calculator as base calculator
cutoff = Gs["default"]["cutoff"]

base_calc = MultiMorse(images, cutoff, combo="mean")


# define learner_params OfflineActiveLearner

learner_params = {
    "atomistic_method": NEBcalc(
        starting_images=images, ml2relax=ml2relax, intermediate_samples=3
    ),
    "max_iterations": 7,
    "samples_to_retrain": 4,
    "filename": "example",
    "file_dir": "./",
    "use_dask": False,
}

learner = OfflineActiveLearner(learner_params, trainer, images, parent_calc, base_calc)
learner.learn()
