import torch
from al_mlp.preset_learners.neb_learner import NEBLearner
import numpy as np
from al_mlp.atomistic_methods import NEBcalc
from al_mlp.base_calcs.morse import MultiMorse
from amptorch.trainer import AtomsTrainer
from torch.nn import Tanh


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

    # define learner_params OfflineActiveLearner

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

    learner = NEBLearner(learner_params, trainer, images, parent_calc, base_calc)
    learner.learn()
    return learner
