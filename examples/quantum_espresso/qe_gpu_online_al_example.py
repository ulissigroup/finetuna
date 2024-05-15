# This is a script running Finetuna with GPU version of Quantum Espresso (QE).
# This script uses QE build hosted on https://catalog.ngc.nvidia.com/orgs/hpc/containers/quantum_espresso,
# openmpi 4.1.2, gcc 11.1.0 and singularity 3.5.3 were used to start QE.
# This script runs a relaxation for a system with CH3 on Cu 3*3*3 slab.

from finetuna.atomistic_methods import Relaxation
from finetuna.online_learner.online_learner import OnlineLearner
from ase.optimize import BFGS
from finetuna.ml_potentials.finetuner_calc import FinetunerCalc
from ase.io import Trajectory
from finetuna.utils import calculate_surface_k_points
from ase.calculators.socketio import SocketIOCalculator
from ase.calculators.espresso import Espresso
import sys
import os

if __name__ == "__main__":
    traj = Trajectory("ch3_cu_final.traj")  # change this path to your trajectory file

    ml_potential = FinetunerCalc(
        checkpoint_path="/home/jovyan/shared-scratch/ocp_checkpoints/for_finetuna/public_checkpoints/scaling_attached/gemnet_t_direct_h512_all_attscale.pt",  # change this path to your gemnet checkpoint,
        mlp_params={
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
            "optim": {
                "batch_size": 1,
                "num_workers": 0,
                "max_epochs": 400,
                "lr_initial": 0.0003,
                "factor": 0.9,
                "eval_every": 1,
                "patience": 3,
                "checkpoint_every": 100000,
                "scheduler_loss": "train",
                "weight_decay": 0,
                "eps": 1e-8,
                "optimizer_params": {
                    "weight_decay": 0,
                    "eps": 1e-8,
                },
            },
            "task": {
                "primary_metric": "loss",
            },
        },
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
        "Cu": "Cu.UPF",
        "C": "C.UPF",
        "O": "O.UPF",
        "H": "H.UPF",
    }

    unixsocket = "unix"

    PWD = os.getcwd()

    gpu_num = 4  # change this to match the number of gpus on your machine

    command = f"mpirun -np {gpu_num} /opt/qe-7.0/bin/pw.x -in espresso.pwi --ipi {unixsocket}:UNIX > espresso.pwo"
    os.environ[
        "ESPRESSO_PSEUDO"
    ] = "/home/jovyan/working/espresso_stuff/gbrv_pseudopotentials"  # set the pseudopotential directory env variable

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
                "query_every_n_steps": 100,
                "num_initial_points": 1,
                "fmax_verify_threshold": 0.03,
                "wandb_init": {
                    "wandb_log": True,
                },
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
