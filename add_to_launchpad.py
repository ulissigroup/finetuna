from fireworks.user_objects.dupefinders.dupefinder_exact import DupeFinderExact
from fireworks import Firework, LaunchPad, FWorker, Workflow
from fireworks.features.multi_launcher import launch_multiprocess
from fireworks.core.rocket_launcher import rapidfire
from al_mlp.online_learner.online_learner_task import OnlineLearnerTask
from al_mlp.ml_potentials.amptorch_ensemble_calc import AmptorchEnsembleCalc
from ase.calculators.vasp import Vasp
import numpy as np
from al_mlp.atomistic_methods import Relaxation
from amptorch.trainer import AtomsTrainer
from ase.io import Trajectory
from ase.io.trajectory import TrajectoryWriter
from al_mlp.online_learner.online_learner import OnlineLearner
from ase.optimize import BFGS
import torch
import os
import copy
import jsonpickle
from utilities import extract_job_parameters
#from vasp_interactive import VaspInteractive


if __name__ == "__main__":


    structures = [Trajectory('structures/MgO_init_structure.traj')[0].copy() for i in range(10)]
    for i, structure in enumerate(structures):
        structure.rattle(0.1, seed=np.random.randint(0,high=100000))
        writer = TrajectoryWriter(filename=f"structures/MgO_init_structure_{i}.traj", mode='w', atoms=structure)
        writer.write()


    breakpoint()
    job_id = int(os.environ['JOB_ID']) # should be unique ID
    host_id = os.environ['HOSTNAME']

    params = extract_job_parameters(job_id)
    cores=16
    # Unpack the params to variables
    num_layers = params['num_layers']
    num_nodes = params['num_nodes']
    stat_uncertain_tol = params['stat_uncertain_tol']
    dyn_uncertain_tol = params['dyn_uncertain_tol']
    maxstep = params['maxstep']
#    cores = params['cores']

    elements = np.unique(np.array(structures[0].get_chemical_symbols())).tolist()
    # Point the launchpad to the remote database on NERSC 
    launchpad = LaunchPad(host='mongodb07.nersc.gov',
                          name='fw_oal',
                          password='gfde223223222rft3',
                          port=27017,
                          username='fw_oal_admin')
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

    parent_calc = Vasp(
        algo="Fast",
        prec="Normal",
#        ibrion=2,  # conjugate gradient descent for relaxations
        isif=0,
        ismear=0,
        ispin=1,  # assume no magnetic moment initially
        ediff=1e-4,
        command=f"mpirun -np {cores} /opt/vasp.6.1.2_pgi_mkl_beef/bin/vasp_std",
        ediffg=-0.03,
        xc="rpbe",
        encut=400,  # planewave cutoff
        lreal=True,  # for slabs lreal is True for bulk False
        nsw=0,  # number of ionic steps in the relaxation
        #                isym=-1,
        lwave=False, # Don't save the WAVECAR for memory reasons
        kpts=(5, 5, 1),
    )

    learner_params = {
        "max_iterations": 10,
        "samples_to_retrain": 1,
        "filename": "relax_example",
        "file_dir": "./",
        "stat_uncertain_tol": stat_uncertain_tol, # eV/A
        "dyn_uncertain_tol": dyn_uncertain_tol, # Just a multiplier
        "fmax_verify_threshold": 0.05,  # eV/AA
        "relative_variance": True,
        "n_ensembles": 10,
        "use_dask": False,
        "parent_calc": parent_calc,
        "optim_relaxer": BFGS,
        "f_max": 0.05,
        "steps": 100,
        "maxstep": maxstep, # Might need larger time step
        "ml_potential": AmptorchEnsembleCalc,
    }

    config = {
        "model": {"get_forces": True, "num_layers": num_layers, "num_nodes": num_nodes},
        "optim": {
            "device": "cpu",
            "force_coefficient": 20.0,
            "lr": 1,
            "batch_size": 10,
            "epochs": 100,
            "optimizer": torch.optim.LBFGS,
            "optimizer_args": {"optimizer__line_search_fn": "strong_wolfe"},
        },
        "dataset": {
            "raw_data": [],
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

    # We need to encode all the configs before passing them as fw_spec. This is because fireworks
    # only handles serialization for primitives and not for custom class objects, which we currently
    # define as part of our configs

    trainer_config_encoded = jsonpickle.encode(config)
    learner_params_encoded = jsonpickle.encode(learner_params)

    for i, structure in enumerate(structures):
         
        filename = f"MgO_relaxation_{i}"
        fireworks = [Firework(
            OnlineLearnerTask(),
            spec={
                "learner_params": learner_params_encoded,
                "trainer_config": trainer_config_encoded,
                "parent_dataset": "/home/jovyan/al_mlp_repo/images.traj",
                "filename": filename,
                "init_structure_path": f"/home/jovyan/al_mlp_repo/structures/MgO_init_structure_{i}.traj",
                "task_name": f"OAL_MgO_{stat_uncertain_tol}_{host_id}",
                "scheduler_file": '/tmp/my-scheduler.json',
                "_add_launchpad_and_fw_id": True,
                #"_dupefinder": DupeFinderExact() # to prevent re-running jobs with duplicate specs!
                },

            name=f"OAL_MgO_{stat_uncertain_tol}_unc_tol",
        )]

        wf = Workflow(fireworks)
        launchpad.add_wf(wf)
