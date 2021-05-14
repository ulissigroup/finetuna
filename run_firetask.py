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
from al_mlp.online_learner.online_learner import OnlineLearner
from ase.optimize import BFGS
import torch
import os
import copy
import jsonpickle
import yaml
import pprint

def extract_job_parameters(job_id):
    """This function is used to extract the parameter set from the master job_params.yml
    file."""
    with open("job_params.yml", 'r') as stream:
        hyper_param_set = yaml.safe_load(stream)
    print("\nParameter set for job_id: ",job_id)
    print("------------------------------------")
    pprint.pprint(hyper_param_set[job_id-1]["param_set"])
    print("------------------------------------\n")
    return hyper_param_set[job_id-1]["param_set"]



if __name__ == "__main__":

    job_id = int(os.environ['JOB_ID']) # should be unique ID

    params = extract_job_parameters(job_id)

    # Unpack the params to variables
    uncertain_tol = params['uncertain_tol']
    cores = params['cores']
    # Set the environment variables for VASP
    os.environ[
        "VASP_COMMAND"
    ] = f"mpirun -np {cores} /opt/vasp.6.1.2_pgi_mkl_beef/bin/vasp_std"
    #breakpoint()
    launchpad = LaunchPad(host='mongodb07.nersc.gov',
                          name='fw_oal',
                          password='gfde223223222rft3',
                          port=27017,
                          username='fw_oal_admin')
#    launchpad.reset('', require_password=False)
    # import make_ensemble and dask for setting parallelization
#    from dask.distributed import Client, LocalCluster
#
#    cluster = LocalCluster(processes=True, n_workers=10, threads_per_worker=1)
#    client = Client(cluster)
#    AmptorchEnsembleCalc.set_executor(client)
    elements = ["Mg", "O"]

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
        prec="Accurate",
        ibrion=2,  # conjugate gradient descent for relaxations
        isif=0,
        ismear=0,
        ispin=1,  # assume no magnetic moment initially
        ediff=1e-4,
        ediffg=-0.05,
        xc="rpbe",
        encut=500,  # planewave cutoff
        lreal=True,  # for slabs lreal is True for bulk False
        nsw=0,  # number of ionic steps in the relaxation
        #                isym=-1,
        kpts=(5, 4, 1),
    )

    learner_params = {
        "max_iterations": 10,
        "samples_to_retrain": 1,
        "filename": "relax_example",
        "file_dir": "./",
        "uncertain_tol": uncertain_tol,  # Very strict - will do mostly parent calls at the start and gather training data points
        "fmax_verify_threshold": 0.05,  # eV/AA
        "relative_variance": True,
        "n_ensembles": 10,
        "use_dask": False,
        "parent_calc": parent_calc,
        "optim_relaxer": BFGS,
        "f_max": 0.05,
        "steps": 100,
        "maxstep": 0.02,
        "ml_potential": AmptorchEnsembleCalc,
    }

    config = {
        "model": {"get_forces": True, "num_layers": 5, "num_nodes": 5},
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
#    learner_params_encoded = jsonpickle.encode(learner_params)
    filename = "MgO_slab_relaxation"
    #
    #    # Instantiate the Firework made up of one firetask
    # Let's try and tune the uncertain_tol by launching parallel FireWorks

    learner_params_set = [dict(learner_params, uncertain_tol=tol) for tol in [0.6]]
    learner_params_set_encoded = [jsonpickle.encode(lps) for lps in learner_params_set]


    fireworks = [Firework(
        OnlineLearnerTask(),
        spec={
            "learner_params": lpse,
            "trainer_config": trainer_config_encoded,
            "parent_dataset": os.path.join(os.getcwd(), "images.traj"),
            "filename": filename,
            "init_structure_path": os.path.join(
                os.getcwd(), "MgO_init_structure.traj"
            ),  # absolute path of the .traj file containing the initial structure
            #"db_path": "/home/jovyan/atomate/config/db.json",
            "task_name":f"OAL_{lps['uncertain_tol']}_thresh",
            "scheduler_file": '/home/jovyan/my-scheduler.json' },
        name=f"OAL_{lps['uncertain_tol']}_thresh",
    ) for lps,lpse in zip(learner_params_set, learner_params_set_encoded)]

    # Let's try and screen through a hyperparameter like n_ensembles through Fireworks. We will start might just add a set of FWs to the WF and run them
    # "all at once"
    #    learner_params_set = [dict(learner_params,n_ensembles=i) for i in range(5,7)]

    #    fireworks = [Firework(OnlineLearnerTask(),
    #        spec={'learner_params': jsonpickle.encode(learner_params),
    #            'trainer_config': trainer_config_encoded,
    #            'parent_dataset': os.path.join(os.getcwd(), 'images.traj'),
    #            'filename': filename,
    #            'init_structure_path': os.path.join(os.getcwd(), init_struct_filename),# absolute path of the .traj file containing the initial structure
    #            'db_path':'/home/jovyan/atomate/config/db.json'}
    #            ,name=f"OAL_FW_{i+1}") for i,learner_params in enumerate(learner_params_set)]
    #
    wf = Workflow(fireworks)
    launchpad.add_wf(wf)
#    launch_multiprocess(launchpad,
#                         FWorker(),
#                         'DEBUG',
#                         nlaunches=0,
#                         num_jobs=1,
#                         sleep_time=0.5,
#                         ppn=20)
#
    rapidfire(launchpad, FWorker(name="test_kubernetes"))
