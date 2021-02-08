from ase.eos import EquationOfState
from ase.build import bulk
from ase.build import fcc100, add_adsorbate, molecule
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from ase.calculators.emt import EMT
import numpy as np
import copy
import torch


from al_mlp.preset_learners.ensemble_learner import EnsembleLearner
from al_mlp.base_calcs.morse import MultiMorse
from al_mlp.atomistic_methods import Relaxation

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
        "lr": 5e-2,
        "batch_size": 1000,
        "epochs": 200,
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
base_calc = MultiMorse(images, cutoff, combo="mean")


learner_params = {
    "atomistic_method": Relaxation(
        initial_geometry=slab.copy(), optimizer=BFGS, fmax=0.01, steps=50
    ),
    "max_iterations": 10,
    "samples_to_retrain": 5,
    "filename": "relax_example",
    "file_dir": "./",
    "query_method": "max_uncertainty",
    "use_dask": False,
}

learner = EnsembleLearner(
    learner_params, trainer, images, parent_calc, base_calc, ensemble=3
)
learner.learn()
