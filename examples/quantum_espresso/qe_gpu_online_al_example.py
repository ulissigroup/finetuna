# This is a script running Finetuna with GPU version of Quantum Espresso (QE).
# This script uses QE build hosted on https://catalog.ngc.nvidia.com/orgs/hpc/containers/quantum_espresso,
# openmpi 4.1.2, gcc 11.1.0 and singularity 3.5.3 were used to start QE.
# This script runs a relaxation for a system with CH3 on Cu 3*3*3 slab.

from finetuna.atomistic_methods import Relaxation
from finetuna.online_learner.online_learner import OnlineLearner
from ase.optimize import BFGS
from finetuna.ml_potentials.finetuner_ensemble_calc import FinetunerEnsembleCalc
from ase.io import Trajectory
from finetuna.utils import calculate_surface_k_points
from ase.calculators.socketio import SocketIOCalculator
from ase.calculators.espresso import Espresso
import sys
import os

if __name__ == "__main__":

    traj = Trajectory(
        "/work/westgroup/chao/qm_calc/new_cpox_dfts/adsorption/ch3_cu/oncv/ch3_cu_final.traj"
    )  # change this path to your trajectory file

    ml_potential = FinetunerEnsembleCalc(
        checkpoint_paths=[
            "/work/westgroup/chao/finetuna_models/s2ef/all/gemnet_t_direct_h512_all.pt",  # change this path to your gemnet checkpoint
        ],
        mlp_params=[
            {
                "tuner": {
                    "unfreeze_blocks": [
                        "out_blocks.3.seq_forces",
                        "out_blocks.3.scale_rbf_F",
                        "out_blocks.3.dense_rbf_F",
                        "out_blocks.3.out_forces",
                        "out_blocks.2.seq_forces",
                        "out_blocks.2.scale_rbf_F",
                        "out_blocks.2.dense_rbf_F",
                        "out_blocks.2.out_forces",
                        "out_blocks.1.seq_forces",
                        "out_blocks.1.scale_rbf_F",
                        "out_blocks.1.dense_rbf_F",
                        "out_blocks.1.out_forces",
                    ],
                    "num_threads": 8,
                },
                "trainer": "forces",
                "dataset": {
                    "src": "/work/westgroup/opencatalyst/ocp/data/s2ef/all/train/",  # change this to your database directory
                    "normalize_labels": True,
                    "target_mean": -0.7554450631141663,
                    "target_std": 2.887317180633545,
                    "grad_target_mean": 0.0,
                    "grad_target_std": 2.887317180633545,
                },
                "logger": "tensorboard",
                "task": {
                    "dataset": "trajectory_lmdb",
                    "description": "Regressing to energies and forces for DFT trajectories from OCP",
                    "type": "regression",
                    "metric": "mae",
                    "labels": ["potential energy"],
                    "grad_input": "atomic forces",
                    "train_on_free_atoms": True,
                    "eval_on_free_atoms": True,
                },
                "model": {
                    "name": "gemnet_t",
                    "num_spherical": 7,
                    "num_radial": 128,
                    "num_blocks": 3,
                    "emb_size_atom": 512,
                    "emb_size_edge": 512,
                    "emb_size_trip": 64,
                    "emb_size_rbf": 16,
                    "emb_size_cbf": 16,
                    "emb_size_bil_trip": 64,
                    "num_before_skip": 1,
                    "num_after_skip": 2,
                    "num_concat": 1,
                    "num_atom": 3,
                    "cutoff": 6.0,
                    "max_neighbors": 50,
                    "rbf": {
                        "name": "gaussian",
                    },
                    "envelope": {
                        "name": "polynomial",
                        "exponent": 5,
                    },
                    "cbf": {
                        "name": "spherical_harmonics",
                    },
                    "extensive": True,
                    "otf_graph": False,
                    "output_init": "HeOrthogonal",
                    "activation": "silu",
                    "scale_file": "/work/westgroup/opencatalyst/ocp/configs/s2ef/all/gemnet/scaling_factors/gemnet-dT.json",  # change this to your scaling file directory
                    "regress_forces": True,
                    "direct_forces": True,
                },
                "optim": {
                    "batch_size": 32,
                    "eval_batch_size": 32,
                    "eval_every": 5000,
                    "num_workers": 2,
                    "lr_initial": 5.0e-4,
                    "optimizer": "AdamW",
                    "optimizer_params": {"amsgrad": True},
                    "scheduler": "ReduceLROnPlateau",
                    "mode": "min",
                    "factor": 0.8,
                    "patience": 3,
                    "max_epochs": 80,
                    "force_coefficient": 100,
                    "energy_coefficient": 1,
                    "ema_decay": 0.999,
                    "clip_grad_norm": 10,
                    "loss_energy": "mae",
                    "loss_force": "l2mae",
                },
            },
        ],
    )

    # change this to your QE settings
    espresso_settings = {
        "control": {
            "verbosity": "high",
            "calculation": "scf",
        },
        "system": {
            "input_dft": "BEEF-VDW",
            "occupations": "smearing",
            "smearing": "mv",
            "degauss": 0.01,
            "ecutwfc": 60,
        },
        "electrons": {
            "electron_maxstep": 200,
            "mixing_mode": "local-TF",
            "conv_thr": 1e-8,
        },
    }

    pseudopotentials = {
        "Cu": "Cu_ONCV_PBE-1.2.upf",
        "C": "C_ONCV_PBE-1.2.upf",
        "O": "O_ONCV_PBE-1.2.upf",
        "H": "H_ONCV_PBE-1.2.upf",
    }

    unixsocket = "unix"

    PWD = os.getcwd()

    gpu_num = 4  # change this to match the number of gpus on your machine

    command = f"singularity run --nv -B{PWD}:/host_pwd --pwd /host_pwd docker://nvcr.io/hpc/quantum_espresso:qe-7.0 mpirun -np {gpu_num} pw.x -in espresso.pwi --ipi {unixsocket}:UNIX > espresso.pwo"

    espresso = Espresso(
        command=command,
        pseudopotentials=pseudopotentials,
        tstress=True,
        tprnfor=True,
        kpts=(4, 4, 1),
        input_data=espresso_settings,
    )

    with SocketIOCalculator(
        espresso, log=sys.stdout, unixsocket=unixsocket
    ) as parent_calc:
        learner = OnlineLearner(
            learner_params={
                "stat_uncertain_tol": 1000000,
                "dyn_uncertain_tol": 1000000,
                "dyn_avg_steps": 15,
                "query_every_n_steps": 100,
                "num_initial_points": 0,
                "initial_points_to_keep": [],
                "fmax_verify_threshold": 0.03,
                "tolerance_selection": "min",
                "partial_fit": True,
            },
            parent_dataset=[],
            ml_potential=ml_potential,
            parent_calc=parent_calc,
            mongo_db=None,
            optional_config=None,
        )

        relaxer = Relaxation(
            initial_geometry=traj[0], optimizer=BFGS, fmax=0.03, steps=None, maxstep=0.2
        )
        relaxer.run(
            calc=learner,
            filename="online_learner_trajectory",
            replay_traj="parent_only",
            max_parent_calls=None,
            check_final=False,
            online_ml_fmax=learner.fmax_verify_threshold,
        )

    print("done!")
