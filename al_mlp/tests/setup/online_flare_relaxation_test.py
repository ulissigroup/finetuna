import numpy as np
from al_mlp.online_learner.online_learner import OnlineLearner
from al_mlp.ml_potentials.flare_pp_calc import FlarePPCalc
import os


def run_online_al(atomistic_method, images, elements, dbname, parent_calc):
    wandb_init = {"wandb_log": False, "name": "unittest", "group": "unittest"}
    learner_params = {
        "max_iterations": 10,
        "samples_to_retrain": 1,
        "filename": "relax_example",
        "file_dir": "./",
        "stat_uncertain_tol": 0.1,
        "dyn_uncertain_tol": 0.1,
        "fmax_verify_threshold": 0.05,  # eV/AA
        "relative_variance": True,
        "n_ensembles": 10,
        "use_dask": True,
        "wandb_init": wandb_init,
    }

    flare_params = {
        "sigma": 1.0,
        "power": 2,
        "cutoff_function": "quadratic",
        "cutoff": 3.0,
        "radial_basis": "chebyshev",
        "cutoff_hyps": [],
        "sigma_e": 0.01,
        "sigma_f": 0.1,
        "sigma_s": 0.0,
        "hpo_max_iterations": 50,
    }
    ml_potential = FlarePPCalc(flare_params, images)

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
