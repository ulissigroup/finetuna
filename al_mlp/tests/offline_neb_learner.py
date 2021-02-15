import torch
from al_mlp.preset_learners.neb_learner import NEBLearner
import numpy as np
from offline_neb_Cu_C_utils import construct_geometries
from al_mlp.atomistic_methods import NEBcalc
from al_mlp.base_calcs.morse import MultiMorse
from amptorch.trainer import AtomsTrainer
from torch.nn import Tanh


def offline_neb(parent_calc, iter=4, intermediate_images=3):
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

    ml2relax = True  # use machine learning to relax the initial and final states rather than DFT as is the norm
    initial, final = construct_geometries(parent_calc=parent_calc, ml2relax=ml2relax)
    images = [initial]
    images.append(final)

    elements = ["Cu", "C"]
    config = {
        "model": {"get_forces": True, "num_layers": 3, "num_nodes": 20, "activation": Tanh},
        "optim": {
            "device": "cpu",
            "force_coefficient": 20,
            "lr": 1e-2,
            "batch_size": 1000,
            "epochs": 500,
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
    # base_calc = Dummy(images)

    # define learner_params OfflineActiveLearner

    learner_params = {
        "atomistic_method": NEBcalc(
            starting_images=images,
            ml2relax=ml2relax,
            intermediate_samples=intermediate_images,
        ),
        "max_iterations": iter,
        "samples_to_retrain": intermediate_images,
        "filename": "example",
        "file_dir": "./",
        "use_dask": False,
        "max_evA": 0.01,
    }

    learner = NEBLearner(learner_params, trainer, images, parent_calc, base_calc)
    learner.learn()

    return learner
