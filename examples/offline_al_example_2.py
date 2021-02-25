from ase.eos import EquationOfState
from ase.build import bulk
from ase.build import fcc100, add_adsorbate, molecule
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from ase.calculators.emt import EMT
import ase.io
import numpy as np
import copy

import torch
import ase

from ase.db import connect
from al_mlp.offline_learner import OfflineActiveLearner
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

# slab = ase.io.read("slab2.traj")
slab.set_pbc(True)
slab.wrap(pbc=[True] * 3)
slab.set_calculator(copy.copy(parent_calc))
slab.set_initial_magnetic_moments()
db = connect("relax_example.db")
images = [slab]

Gs = {
    "default": {
        "G2": {
            "etas": np.logspace(np.log10(0.05), np.log10(5.0), num=4),
            "rs_s": [0] * 4,
        },
        "G4": {"etas": [0.005], "zetas": [1.0, 4.0], "gammas": [1.0, -1.0]},
        "cutoff": 6,
    },
}

elements = np.unique(images[0].get_chemical_symbols())

learner_params = {
    "atomistic_method": Relaxation(
        initial_geometry=slab.copy(), optimizer=BFGS, fmax=0.01, steps=100
    ),
    "max_iterations": 10,
    "force_tolerance": 0.01,
    "samples_to_retrain": 3,
    "filename": "relax_example",
    "file_dir": "./",
    "query_method": "random",
    "use_dask": False,
    "max_evA": 0.05,
}

config = {
    "model": {
        "get_forces": True,
        "num_layers": 3,
        "num_nodes": 20,
    },
    "optim": {
        "device": "cpu",
        "force_coefficient": 40,
        "lr": 0.1,
        "batch_size": 100,
        "epochs": 200,  # was 100
        "loss": "mse",
        "metric": "mae",
        "optimizer": torch.optim.LBFGS,
        "optimizer_args": {"optimizer__line_search_fn": "strong_wolfe"},
    },
    "dataset": {
        "raw_data": images,
        "val_split": 0,
        "elements": elements,
        "fp_params": Gs,
        "save_fps": False,
        "scaling": {"type": "normalize", "range": (-1, 1)},
    },
    "cmd": {
        "debug": False,
        "run_dir": "./",
        "seed": 1,
        "identifier": "test",
        "verbose": True,
        # "logger": True,
        "single-threaded": True,
    },
}

trainer = AtomsTrainer(config)
# building base morse calculator as base calculator
cutoff = Gs["default"]["cutoff"]
base_calc = MultiMorse(images, cutoff, combo="mean")

learner = OfflineActiveLearner(
    learner_params,
    trainer,
    images,
    parent_calc,
    base_calc,
)

learner.learn()

# Calculate true relaxation
al_iterations = learner.iterations - 1
file_path = learner_params["file_dir"] + learner_params["filename"]
true_relax = Relaxation(slab, BFGS, fmax=0.01)
true_relax.run(EMT(), "true_relax")
parent_calc_traj = true_relax.get_trajectory("true_relax")
final_ml_traj = ase.io.read("{}_iter_{}.traj".format(file_path, al_iterations), ":")

# Compute ML predicted energies
ml_relaxation_energies = [image.get_potential_energy() for image in final_ml_traj]
# Compute actual (EMT) energies for ML predicted structures
emt_evaluated_ml_energies = [
    EMT().get_potential_energy(image) for image in final_ml_traj
]
# Compute actual energies for EMT relaxation structures
emt_relaxation_energies = [image.get_potential_energy() for image in parent_calc_traj]
steps = range(len(final_ml_traj))
n_samples_iteration = learner_params["samples_to_retrain"]
parent_calls = learner.parent_calls


def compute_loss(a, b):
    return np.mean(np.sqrt(np.sum((a - b) ** 2, axis=1)))


initial_structure = images[0].positions
print(f"Number of AL iterations: {al_iterations}")
print(f"Number of samples/iteration: {n_samples_iteration}")
print(f"Total # of queries (parent calls): {parent_calls}\n")

print(f"Final AL Relaxed Energy: {ml_relaxation_energies[-1]}")
print(
    f"EMT evaluation at AL structure: {EMT().get_potential_energy(final_ml_traj[-1])}\n"
)
al_relaxed_structure = final_ml_traj[-1].positions

print(f"Total number of EMT steps: {len(emt_relaxation_energies)}")
print(f"Final EMT Relaxed Energy: {emt_relaxation_energies[-1]}\n")
emt_relaxed_structure = parent_calc_traj[-1].positions

initial_structure_error = compute_loss(initial_structure, emt_relaxed_structure)
relaxed_structure_error = compute_loss(al_relaxed_structure, emt_relaxed_structure)

print(f"Initial structure error: {initial_structure_error}")
print(f"AL relaxed structure error: {relaxed_structure_error}")
