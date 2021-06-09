import numpy as np
from al_mlp.online_learner.online_learner import OnlineLearner
from al_mlp.ml_potentials.amptorch_ensemble_calc import AmptorchEnsembleCalc

from amptorch.trainer import AtomsTrainer
import os
import torch


def run_online_al(atomistic_method, images, elements, dbname, parent_calc):

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

    learner_params = {
        "max_iterations": 10,
        "samples_to_retrain": 1,
        "filename": "relax_example",
        "file_dir": "./",
        "stat_uncertain_tol": 0.15,
        "dyn_uncertain_tol": 1.5,
        "fmax_verify_threshold": 0.05,  # eV/AA
        "relative_variance": True,
        "n_ensembles": 5,
        "use_dask": True,
    }

    config = {
        "model": {"get_forces": True, "num_layers": 3, "num_nodes": 5},
        "optim": {
            "device": "cpu",
            "force_coefficient": 4.0,
            "lr": 1e-2,
            "batch_size": 10,
            "epochs": 100,
            "optimizer": torch.optim.LBFGS,
            "optimizer_args": {"optimizer__line_search_fn": "strong_wolfe"},
        },
        "dataset": {
            "raw_data": images,
            "val_split": 0,
            "elements": elements,
            "fp_params": Gs,
            "save_fps": False,
            "scaling": {"type": "standardize"},
        },
        "cmd": {
            "debug": False,
            "run_dir": "./",
            "seed": 1,
            "identifier": "test",
            "verbose": False,
            # "logger": True,
            "single-threaded": True,
        },
    }

    trainer = AtomsTrainer(config)

    ml_potential = AmptorchEnsembleCalc(trainer, learner_params["n_ensembles"])
    onlinecalc = OnlineLearner(
        learner_params,
        images,
        ml_potential,
        parent_calc,
    )

    if os.path.exists("dft_calls.db"):
        os.remove("dft_calls.db")
    atomistic_method.run(onlinecalc, filename=dbname)

    return onlinecalc, atomistic_method
