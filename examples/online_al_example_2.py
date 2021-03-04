from al_mlp.atomistic_methods import Relaxation
from al_mlp.online_learner import OnlineLearner
from amptorch.trainer import AtomsTrainer
from ase.calculators.emt import EMT
import numpy as np
from ase.cluster.icosahedron import Icosahedron
from ase.optimize import BFGS
import torch
import os
import copy

# Set up ensemble parallelization
if __name__ == "__main__":
    # import make_ensemble and dask for setting parallelization
    from al_mlp.ensemble_calc import EnsembleCalc
    from dask.distributed import Client, LocalCluster

    cluster = LocalCluster(processes=True, threads_per_worker=1)
    client = Client(cluster)
    EnsembleCalc.set_executor(client)

    # Set up parent calculator and image environment
    initial_structure = Icosahedron("Cu", 2)
    initial_structure.rattle(0.1)
    initial_structure.set_pbc(True)
    initial_structure.set_cell([20, 20, 20])
    images = []
    elements = ["Cu"]
    parent_calc = EMT()

    # Run relaxation with active learning
    OAL_initial_structure = initial_structure.copy()
    OAL_initial_structure.set_calculator(copy.deepcopy(parent_calc))
    OAL_relaxation = Relaxation(
        OAL_initial_structure, BFGS, fmax=0.05, steps=60, maxstep=0.04
    )

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
        "uncertain_tol": 5.0,
        "fmax_verify_threshold": 0.05,  # eV/AA
        "relative_variance": True,
        "n_ensembles": 10,
        "use_dask": True,
    }

    config = {
        "model": {"get_forces": True, "num_layers": 3, "num_nodes": 5},
        "optim": {
            "device": "cpu",
            "force_coefficient": 4.0,
            "lr": 1,
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

    dbname = "CuNP_oal"
    trainer = AtomsTrainer(config)

    onlinecalc = OnlineLearner(
        learner_params,
        trainer,
        images,
        copy.deepcopy(parent_calc),
    )

    if os.path.exists("dft_calls.db"):
        os.remove("dft_calls.db")
    OAL_relaxation.run(onlinecalc, filename=dbname)

    # Retain and print image of the final structure from the online relaxation
    OAL_image = OAL_relaxation.get_trajectory("CuNP_oal")[-1]
    OAL_image.set_calculator(copy.deepcopy(parent_calc))
    print(
        "Final Image Results:"
        + "\nEnergy:\n"
        + str(OAL_image.get_potential_energy())
        + "\nForces:\n"
        + str(OAL_image.get_forces())
    )
