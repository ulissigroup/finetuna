import torch
from al_mlp.offline_learner.neb_learner import NEBLearner
import numpy as np
from al_mlp.atomistic_methods import NEBcalc
from al_mlp.base_calcs.morse import MultiMorse
from amptorch.trainer import AtomsTrainer
from torch.nn import Tanh

from ase.optimize import BFGS
from ase.io import read
from ase.build import fcc100, add_adsorbate
from ase.constraints import FixAtoms


def offline_neb(images, parent_calc, iter=4, intermediate_images=3):
    torch.set_num_threads(1)

    parent_calc = parent_calc

    Gs = {
        "default": {
            "G2": {
                "etas": np.logspace(np.log10(0.05), np.log10(5.0), num=4),
                "rs_s": [0],
            },
            "G4": {"etas": [0.005], "zetas": [1.0, 4.0], "gammas": [1.0, -1.0]},
            "cutoff": 5.0,
        },
    }

    elements = ["Cu", "C"]
    config = {
        "model": {
            "get_forces": True,
            "num_layers": 3,
            "num_nodes": 20,
            "activation": Tanh,
        },
        "optim": {
            "device": "cpu",
            "force_coefficient": 27,
            "lr": 1e-2,
            "batch_size": 1000,
            "epochs": 300,
            "loss": "mse",
            "metric": "mse",
            "optimizer": torch.optim.LBFGS,
            "optimizer_args": {"optimizer__line_search_fn": "strong_wolfe"},
            "scheduler": {
                "policy": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
                "params": {"T_0": 10, "T_mult": 2},
            },
        },
        "dataset": {
            "raw_data": images,
            "val_split": 0,
            "elements": elements,
            "fp_params": Gs,
            "save_fps": True,
            "scaling": {"type": "normalize", "range": (-1, 1)},
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
    neb_images = images.copy()
    base_calc = MultiMorse(neb_images, cutoff, combo="mean")
    # base_calc = Dummy(images)

    # define learner_params OfflineDeltaLearner

    learner_params = {
        "atomistic_method": NEBcalc(
            starting_images=neb_images,
            intermediate_samples=intermediate_images,
        ),
        "max_iterations": iter,
        "samples_to_retrain": intermediate_images,
        "filename": "example",
        "file_dir": "./",
        "use_dask": False,
        # "max_evA": 0.01,
    }

    learner = NEBLearner(learner_params, images, trainer, parent_calc, base_calc)
    learner.learn()
    return learner


def construct_geometries(parent_calc, ml2relax):
    counter_calc = parent_calc
    # Initial structure guess
    initial_slab = fcc100("Cu", size=(2, 2, 3))
    add_adsorbate(initial_slab, "C", 1.7, "hollow")
    initial_slab.center(axis=2, vacuum=4.0)
    mask = [atom.tag > 1 for atom in initial_slab]
    initial_slab.set_constraint(FixAtoms(mask=mask))
    initial_slab.set_pbc(True)
    initial_slab.wrap(pbc=[True] * 3)
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
        # If there is already a pre-existing initial and final relaxed parent state we can read
        # that to use as a starting point
        # initial_slab = read("/content/parent_initial.traj")
        # final_slab = read("/content/parent_final.traj")
    else:
        initial_slab = initial_slab
        final_slab = final_slab

    # initial_force_calls = counter_calc.force_calls
    return initial_slab, final_slab  # , initial_force_calls
