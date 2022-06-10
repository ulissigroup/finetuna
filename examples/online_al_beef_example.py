from finetuna.atomistic_methods import Relaxation
from finetuna.online_learner.online_learner import OnlineLearner
from vasp_interactive import VaspInteractive
from ase.optimize import BFGS
from finetuna.ml_potentials.finetuner_ensemble_calc import FinetunerEnsembleCalc
from pymatgen.io.vasp.inputs import Kpoints, Incar
from ase.io import read


if __name__ == "__main__":
    incar_path = "/home/jovyan/working/data/richard_ipa/adslabs_inputs/Ag_111_IPA/INCAR"
    kpoints_path = (
        "/home/jovyan/working/data/richard_ipa/adslabs_inputs/Ag_111_IPA/KPOINTS"
    )
    poscar_path = (
        "/home/jovyan/working/data/richard_ipa/adslabs_inputs/Ag_111_IPA/POSCAR"
    )

    slab = read(poscar_path)

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

    incar = Incar.from_file(incar_path)
    kpoints = Kpoints.from_file(kpoints_path)
    vasp_inputs = {
        "setups": "minimal",
        "lbeefens": True,
        "xc": "beef-vdw",
        "kpts": kpoints.kpts[0],
    }
    for k in incar.keys():
        vasp_inputs[k.lower()] = incar[k]
    vasp_inputs["nsw"] = 2000
    vasp_inputs["ibrion"] = -1

    with VaspInteractive(**vasp_inputs) as parent_calc:

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
                # "wandb_init": {
                #     "wandb_log": True,
                #     "project": "10_richard_ipa",
                #     "name": "test0",
                #     "notes": "2022_03_22, test description",
                #     "group": "tests"
                # },
            },
            parent_dataset=[],
            ml_potential=ml_potential,
            parent_calc=parent_calc,
            mongo_db=None,
            optional_config=None,
        )

        relaxer = Relaxation(
            initial_geometry=slab, optimizer=BFGS, fmax=0.03, steps=None, maxstep=0.2
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
