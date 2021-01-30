import numpy as np
import ase
import copy
from al_mlp.online_learner import OnlineActiveLearner
from ase.calculators.emt import EMT
from al_mlp.base_calcs.morse import MultiMorse
from ase import Atoms
from amptorch.ase_utils import AMPtorch
from amptorch.trainer import AtomsTrainer
from ase.build import fcc100, add_adsorbate, molecule
from ase.constraints import FixAtoms
from ase.optimize import BFGS, QuasiNewton
from ase.build import bulk
from ase.utils.eos import EquationOfState

parent_calculator = EMT()
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
add_adsorbate(slab, ads, 2, offset=(1, 1))
cons = FixAtoms(indices=[atom.index for atom in slab if (atom.tag == 3)])
slab.set_constraint(cons)
slab.center(vacuum=13.0, axis=2)
slab.set_pbc(True)
slab.wrap(pbc=[True] * 3)
slab.set_calculator(copy.copy(parent_calculator))
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
learner_params = {
    "max_iterations": 10,
    "samples_to_retrain": 1,
    "filename": "relax_example",
    "file_dir": "./",
    "uncertain_tol": 0.05,
}

config = {
    "model": {"get_forces": True, "num_layers": 3, "num_nodes": 5},
    "optim": {
        "device": "cpu",
        "force_coefficient": 0.04,
        "lr": 1e-2,
        "batch_size": 10,
        "epochs": 100,
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
        # "logger": True,
    },
}
cutoff = Gs["default"]["cutoff"]
parent_calc = EMT()
trainer = AtomsTrainer(config)
trainer_calc = AMPtorch
base_calc = MultiMorse(images, cutoff, combo="mean")

onlinecalc = OnlineActiveLearner(
    learner_params,
    trainer,
    images,
    parent_calc,
    base_calc,
    trainer_calc,
    n_ensembles=5,
    n_cores="max",
)

structure_optim = Relaxation(slab, BFGS, fmax=0.05, steps=100)

structure_optim.run(onlinecalc, filename="relax_oal")
