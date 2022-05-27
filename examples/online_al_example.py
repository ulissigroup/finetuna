from finetuna.atomistic_methods import Relaxation
from finetuna.online_learner.online_learner import OnlineLearner
from vasp_interactive import VaspInteractive
from ase.optimize import BFGS
from finetuna.ml_potentials.finetuner_ensemble_calc import FinetunerEnsembleCalc
from ase.io import Trajectory
from finetuna.utils import calculate_surface_k_points

if __name__ == "__main__":

    traj = Trajectory(
        "/home/jovyan/working/data/30_randoms_n60/random1447590.traj"
    )  # change this path to your trajectory file

    ml_potential = FinetunerEnsembleCalc(
        model_classes=[
            "gemnet",
        ],
        model_paths=[
            "/home/jovyan/working/ocp/configs/s2ef/all/gemnet/gemnet-dT.yml",  # change this path to your gemnet config
        ],
        checkpoint_paths=[
            "/home/jovyan/shared-scratch/joe/optim_cleaned_checkpoints/gemnet_s2re_bagging_results/gem_homo_run0.pt",  # change this path to your gemnet checkpoint
        ],
        mlp_params=[
            {
                # Change "cpu": False if you want to use GPU training (if available)
                "cpu": True,
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
        ],
    )

    parent_calc = VaspInteractive(
        isif=0,
        isym=0,
        lreal="Auto",
        ediffg=-0.03,
        symprec=1.0e-10,
        encut=350.0,
        laechg=False,
        lcharg=False,
        lwave=False,
        ncore=16,
        gga="RP",
        pp="PBE",
        xc="PBE",
        kpts=calculate_surface_k_points(traj[0]),
        # Uncomment the kpts=(1,1,1) for a very fast test (but not accurate)!
        # kpts=(1,1,1),
    )

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
