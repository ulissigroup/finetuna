import numpy as np
import ase
import copy
from al_mlp.online_learner import OnlineActiveLearner
from ase.calculators.emt import EMT
from ase.calculators.calculator import Calculator, all_changes
from al_mlp.base_calcs.morse import MultiMorse
from ase import Atoms
from amptorch.ase_utils import AMPtorch
from amptorch.trainer import AtomsTrainer
from ase.build import fcc100, add_adsorbate, molecule
from ase.constraints import FixAtoms
from ase.optimize import BFGS, QuasiNewton
from ase.build import bulk
from ase.utils.eos import EquationOfState
from al_mlp.atomistic_methods import Relaxation
import os
from al_mlp.ensemble_calc import EnsembleCalc
from al_mlp.base_calcs.dummy import Dummy
import torch
import uuid


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
        "uncertain_tol": 2,
        "relative_variance": True,
        "use_dask": True,
    }

    config = {
        "model": {"get_forces": True, "num_layers": 3, "num_nodes": 5},
        "optim": {
            "device": "cpu",
            "force_coefficient": 1.0,
            "lr": 1e-2,
            "batch_size": 10,
            "epochs": 100,  # was 100
            "optimizer": torch.optim.LBFGS,
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
            "single-threaded": False,
        },
    }

    if learner_params["use_dask"]:
        from dask.distributed import Client, LocalCluster

        cluster = LocalCluster(n_workers=4, processes=True, threads_per_worker=1)
        client = Client(cluster)
        EnsembleCalc.set_executor(client)

    cutoff = Gs["default"]["cutoff"]
    trainer = AtomsTrainer(config)
    trainer_calc = AMPtorch
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
