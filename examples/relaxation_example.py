from ase.eos import EquationOfState
from ase.build import bulk
from ase.build import fcc100, add_adsorbate, molecule
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from ase.calculators.emt import EMT
from ase import Atoms, Atom
import numpy as np
import copy
import sys
import random
import torch

torch.set_num_threads(1)

from al_mlp.offline_active_learner import OfflineActiveLearner
from al_mlp.base_calcs import MultiMorse
from al_mlp.atomistic_methods import Relaxation

from amptorch.ase_utils import AMPtorch
from amptorch.trainer import AtomsTrainer


parent_calc = EMT()
# Make a simple C on Cu slab.
# Sets calculator to parent_calc.

energies = []
volumes = []
LC = [3.5, 3.55, 3.6, 3.65, 3.7, 3.75]

for a in LC:
    cu_bulk = bulk("Cu", "fcc", a=a)

    calc = EMT()

    cu_bulk.set_calculator(calc)

    e = cu_bulk.get_potential_energy()
    energies.append(e)
    volumes.append(cu_bulk.get_volume())


eos = EquationOfState(volumes, energies)
v0, e0, B = eos.fit()
aref = 3.6
vref = bulk("Cu", "fcc", a=aref).get_volume()

copper_lattice_constant = (v0 / vref) ** (1 / 3) * aref

slab = fcc100("Cu", a=copper_lattice_constant, size=(2, 2, 3))
ads = molecule("C")
add_adsorbate(slab, ads, 3, offset=(1, 1))
cons = FixAtoms(indices=[atom.index for atom in slab if (atom.tag == 3)])
slab.set_constraint(cons)
slab.center(vacuum=13.0, axis=2)
slab.set_pbc(True)
slab.wrap(pbc=[True] * 3)
slab.set_calculator(copy.copy(parent_calc))
slab.set_initial_magnetic_moments()

images = [slab]


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

elements = ["Cu", "C"]
config = {
    "model": {"get_forces": True, "num_layers": 3, "num_nodes": 20},
    "optim": {
        "device": "cpu",
        "force_coefficient": 0.04,
        "lr": 1e-2,
        "batch_size": 1000,
        "epochs": 100,
        "loss": "mse",
        "metric": "mae",
        "optimizer": torch.optim.LBFGS,
    },
    "dataset": {
        "raw_data": images,
        "val_split": 0,
        "elements": elements,
        "fp_params": Gs,
        "save_fps": True,
    },
    "cmd": {
        "debug": False,
        "run_dir": "./",
        "seed": 1,
        "identifier": "test",
        "verbose": True,
        "logger": False,
    },
}

trainer = AtomsTrainer(config)

# building base morse calculator as base calculator
cutoff = Gs["default"]["cutoff"]

params = {
    "C": {"re": 1.285943211152638, "D": 8.928283649952903, "a": 1.8923342241917613},
    "Cu": {"re": 2.2178539292118344, "D": 2.774711971071156, "a": 1.3847642126468944},
}

base_calc = MultiMorse(params, cutoff, combo="mean")


# define learner inheriting from OfflineActiveLearner


class Learner(OfflineActiveLearner):

    # can customize init
    def __init__(
        self, learner_settings, trainer, training_data, parent_calc, base_calc
    ):
        super().__init__(
            learner_settings, trainer, training_data, parent_calc, base_calc
        )

    def make_trainer_calc(self):
        return AMPtorch(self.trainer)

    # can customize termination criteria and query strategy
    def check_terminate(self):
        if self.iterations >= 10:
            return True
        return False

    def query_func(self, sample_candidates):
        random.seed()
        queried_images = random.sample(sample_candidates, 2)
        return queried_images


learner = Learner(None, trainer, images, parent_calc, base_calc)
learner.learn(
    atomistic_method=Relaxation(
        initial_geometry=slab.copy(), optimizer=BFGS, fmax=0.01, steps=100
    )
)
