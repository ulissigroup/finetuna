import numpy as np
from al_mlp.online_learner import OnlineActiveLearner
from amptorch.trainer import AtomsTrainer
import os
from al_mlp.ensemble_calc import EnsembleCalc
from al_mlp.base_calcs.dummy import Dummy
import torch

# from amptorch.ase_utils import AMPtorch


def run_oal(atomistic_method, images, dbname, parent_calc):

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

    elements = np.unique(images[0].get_chemical_symbols())

    learner_params = {
        "max_iterations": 10,
        "samples_to_retrain": 1,
        "filename": "relax_example",
        "file_dir": "./",
        "uncertain_tol": 0.1,
        "relative_variance": True,
        "use_dask": True,
    }

    config = {
        "model": {"get_forces": True, "num_layers": 3, "num_nodes": 5},
        "optim": {
            "device": "cpu",
            "force_coefficient": 4.0,
            "lr": 1,
            "batch_size": 10,
            "epochs": 100,  # was 100
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
            "verbose": True,
            # "logger": True,
            "single-threaded": True,
        },
    }

    if learner_params["use_dask"] and EnsembleCalc.executor is None:
        from dask.distributed import Client, LocalCluster

        cluster = LocalCluster(n_workers=4, processes=True, threads_per_worker=1)
        client = Client(cluster)
        EnsembleCalc.set_executor(client)

    # cutoff = Gs["default"]["cutoff"]
    trainer = AtomsTrainer(config)
    # trainer_calc = AMPtorch
    base_calc = Dummy(images)

    onlinecalc = OnlineActiveLearner(
        learner_params,
        trainer,
        images,
        parent_calc,
        base_calc,
        # trainer_calc,
        n_ensembles=10,
        n_cores="max",
    )

    if os.path.exists("dft_calls.db"):
        os.remove("dft_calls.db")
    atomistic_method.run(onlinecalc, filename=dbname)

    return onlinecalc, atomistic_method
