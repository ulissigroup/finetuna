from al_mlp.atomistic_methods import Relaxation
from al_mlp.online_learner.online_learner import OnlineLearner
from ase.calculators.vasp import Vasp
from vasp_interactive import VaspInteractive
from ase.optimize import BFGS
from al_mlp.ml_potentials.finetuner_ensemble_calc import FinetunerEnsembleCalc
from ase.io import Trajectory
from al_mlp.utils import calculate_surface_k_points

def main():
    """Compare the efficiency of OAL + ASE-Vasp or VaspInteractive (w/ or w/o pause)
    usage: 
    python oal_finetune_vasp_inter.py <ase or vasp-interactive> <pause? 1/0>
    """
    import sys
    calc_type = sys.argv[1].lower()
    if len(sys.argv) > 2:
        pause = int(sys.argv[2])
    else:
        pause = 1
        
    print(f"{calc_type} {pause}")
    
    

    traj = Trajectory(
        "/home/jovyan/shared-scratch/joe/30_randoms/random1447590.traj"
    )  # change this path to your trajectory file

    ml_potential = FinetunerEnsembleCalc(
        model_classes=[
            "gemnet",
            "gemnet",
        ],
        model_paths=[
            "/home/jovyan/data/ocp_vdw_tl/configs/s2ef/all/gemnet/gemnet-dT.yml",  # change this path to your gemnet config
            "/home/jovyan/data/ocp_vdw_tl/configs/s2ef/all/gemnet/gemnet-dT.yml",  # change this path to your gemnet config
        ],
        checkpoint_paths=[
            "/home/jovyan/shared-scratch/joe/optim_cleaned_checkpoints/gemnet_s2re_bagging_results/gem_homo_run0.pt",  # change this path to your gemnet checkpoint
            "/home/jovyan/shared-scratch/joe/optim_cleaned_checkpoints/gemnet_s2re_bagging_results/gem_homo_run1.pt",  # change this path to your other gemnet checkpoint
        ],
        mlp_params=[
            {
                "tuner": {
                    "unfreeze_blocks": [
                        "out_blocks.3.seq_forces",
                        "out_blocks.3.scale_rbf_F",
                        "out_blocks.3.dense_rbf_F",
                        "out_blocks.3.out_forces",
                    ],
                    "validation_split": [0],
                    "num_threads": 8,
                },
                "optim": {
                    "batch_size": 1,
                    "num_workers": 0,
                    "max_epochs": 30,
                    "lr_initial": 0.0003,
                    "factor": 0.95,
                    "eval_every": 1,
                    "patience": 3,
                },
            },
            {
                "tuner": {
                    "unfreeze_blocks": [
                        "out_blocks.2.seq_forces",
                        "out_blocks.2.scale_rbf_F",
                        "out_blocks.2.dense_rbf_F",
                        "out_blocks.2.out_forces",
                    ],
                    "validation_split": [0],
                    "num_threads": 8,
                },
                "optim": {
                    "batch_size": 1,
                    "num_workers": 0,
                    "max_epochs": 30,
                    "lr_initial": 0.0003,
                    "factor": 0.95,
                    "eval_every": 1,
                    "patience": 3,
                },
            },
        ],
    )
    common_params = dict(
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
    )

    if calc_type == "ase":
        parent_calc = Vasp(
            ibrion=-1,
            nsw=0,
            **common_params
        )
    else:
        allow_pause = (pause == 1)
        parent_calc = VaspInteractive(
            allow_mpi_pause=allow_pause,
            **common_params
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

if __name__ == "__main__":
    main()
