from ase.io import Trajectory
from ase.optimize.bfgs import BFGS
import numpy as np
import os
from ase.calculators.vasp import Vasp
from al_mlp.atomistic_methods import Relaxation
from al_mlp.utils import calculate_surface_k_points
from al_mlp.ml_potentials.flare_pp_calc import FlarePPCalc
from pymongo import MongoClient
from al_mlp.online_learner.online_learner import OnlineLearner

# from al_mlp.base_calcs.ocp_model import OCPModel
# from vasp_interactive.vasp_interactive import VaspInteractive

filename = "/home/jovyan/shared-scratch/joe/metal_with_c2_val/C2H1O2/random1608175.traj"
initial_traj = Trajectory(filename)
initial_structure = initial_traj[0]
images = []

mongo_db = None
if "MONGOC" in os.environ:
    mongo_string = os.environ["MONGOC"]
    mongo_db = MongoClient(mongo_db)["al_db"]
else:
    print("no recording to mongo db")

all_params = {
    "vasp": {
        "ibrion": -1,
        "nsw": 0,
        "isif": 0,
        "isym": 0,
        "lreal": "Auto",
        "ediffg": -0.03,
        "symprec": 1e-10,
        "encut": 350.0,
        "laechg": False,
        "lcharg": False,
        "lwave": False,
        "ncore": 4,
        "gga": "RP",
        "pp": "PBE",
        "xc": "PBE",
        "kpts": calculate_surface_k_points(initial_structure),
    },
    "Gs": {
        "default": {
            "G2": {
                "etas": np.logspace(np.log10(0.05), np.log10(5.0), num=4),
                "rs_s": [0],
            },
            "G4": {"etas": [0.005], "zetas": [1.0, 4.0], "gammas": [1.0, -1.0]},
            "cutoff": 6,
        },
    },
    "learner": {
        "max_iterations": 100,  # offline
        "samples_to_retrain": 1,  # offline
        "filename": "relax_example",  # offline
        "file_dir": "./",  # offline
        "stat_uncertain_tol": 0.08,  # online
        "dyn_uncertain_tol": 0.1,  # online
        "fmax_verify_threshold": 0.03,  # online
        "seed": 1,  # offline
    },
    "flare": {
        "sigma": 2,
        "power": 2,
        "cutoff_function": "quadratic",
        "cutoff": 5.0,
        "radial_basis": "chebyshev",
        "cutoff_hyps": [],
        "sigma_e": 0.002,
        "sigma_f": 0.05,
        "sigma_s": 0.0006,
        "max_iterations": 50,
        # "update_gp_mode": "uncertain",
        # "update_gp_range": [5],
        "freeze_hyps": 0,
        "variance_type": "SOR",
        "opt_method": "BFGS",
    },
    "ocp": {
        "checkpoint_path": "/home/jovyan/working/data/spinconv_checkpoints/spinconv-2021-05-18-16-23-44-oc20-force-grid-96-59-checkpoint.pt",
        "model_path": "/home/jovyan/working/ocp-modeling-dev/experimental/zitnick/configs/s2ef/all/spinconv_force59.yml",
    },
    "relaxation": {
        "optimizer": BFGS,
        "fmax": 0.03,
        "steps": 2000,
        "maxstep": 0.04,
        "max_parent_calls": None,
    },
}

dbname = "flare_" + str(initial_structure.get_chemical_formula()) + "_oal"

oal_initial_structure = initial_structure

# declare parent calc
parent_calc = Vasp(**all_params["vasp"])

# declare ml calc
ml_potential = FlarePPCalc(all_params["flare"], [initial_structure] + images)

# declare online learner
learner = OnlineLearner(
    all_params["learner"],
    images,
    ml_potential,
    parent_calc,
    mongo_db=mongo_db,
)

if os.path.exists("dft_calls.db"):
    os.remove("dft_calls.db")

if mongo_db is not None:
    with open("runid.txt", "a") as f:
        f.write(str(learner.mongo_wrapper.run_id) + "\n")

oal_relaxation = Relaxation(
    oal_initial_structure,
    all_params["relaxation"]["optimizer"],
    fmax=all_params["relaxation"]["fmax"],
    steps=all_params["relaxation"]["steps"],
    maxstep=all_params["relaxation"]["maxstep"],
)

oal_relaxation.run(
    learner,
    filename=dbname,
    replay_traj=True,
    max_parent_calls=all_params["relaxation"]["max_parent_calls"],
)

if hasattr(parent_calc, "close"):
    parent_calc.close()
