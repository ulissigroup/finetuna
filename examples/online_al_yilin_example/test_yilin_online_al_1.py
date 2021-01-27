import numpy as np
import ase
import copy
from al_mlp.online_learner import OnlineActiveLearner
from ase.calculators.emt import EMT
from ase.calculators.calculator import Calculator, all_changes
from al_mlp.base_calcs.morse import MultiMorse
from ase import Atoms
from amptorch.ase_utils import AMPtorch
from amptorch.trainer import AtomsTrainer
from ase.build import fcc100, add_adsorbate, molecule
from ase.constraints import FixAtoms
from ase.optimize import BFGS, QuasiNewton
from ase.build import bulk
from ase.utils.eos import EquationOfState
from al_mlp.atomistic_methods import Relaxation
import os
from al_mlp.ensemble_calc import EnsembleCalc
#from dummy.py import Dummy

class Dummy(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, images, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.images = images

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        image = atoms
        natoms = len(image)
        energy = 0.0
        forces = np.zeros((natoms, 3))
        self.results["energy"] = energy
        self.results["forces"] = forces

from concurrent.futures import ThreadPoolExecutor
#executor = ThreadPoolExecutor(8)

#Set up dask
from dask_kubernetes import KubeCluster
from dask.distributed import Client

num_workers = 10
cluster = KubeCluster.from_yaml("/home/jovyan/al_mlp/examples/online_al_yilin_example/dask-worker-cpu-spec.yml")
client = Client(cluster)
cluster.adapt(minimum=num_workers, maximum=num_workers)
executor = client

#client.upload_file('dummy.py')#pass dask workers changed files (TEMPORARY)

EnsembleCalc.set_executor(executor)

#Set up parent calculator and image environment
parent_calculator = EMT()

initial_db = ase.io.read("/home/jovyan/scripts/test_yilin/Pt-init-images.db",":")
images = [initial_db[1]]
images[0].calc = EMT()

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

elements = ["Pt" ]
learner_params = { 
        "max_iterations": 10,
        "samples_to_retrain": 1,
        "filename":"relax_example",
        "file_dir":"./",
        "uncertain_tol": 2,
        "relative_variance": True
        }

config = {
    "model": {"get_forces": True, "num_layers": 3, "num_nodes": 5},
    "optim": {
        "device": "cpu",
        "force_coefficient": 0.04,
        "lr": 1e-2,
        "batch_size": 10,
        "epochs": 100, #was 100
    },
    "dataset": {
        "raw_data": images,
        "val_split": 0,
        "elements": elements,
        "fp_params": Gs,
        "save_fps": True,
    },
    "cmd": {
        "debug": False,
        "run_dir": "./",
        "seed": 1,
        "identifier": "test",
        "verbose": True,
        # "logger": True,
        "single-threaded": False,
    },
}
cutoff = Gs["default"]["cutoff"]
parent_calc = EMT()
trainer = AtomsTrainer(config)
trainer_calc = AMPtorch
#base_calc = MultiMorse(images, cutoff, combo="mean")
base_calc = Dummy(images)

onlinecalc = OnlineActiveLearner(
             learner_params,
             trainer,
             images,
             parent_calc,
             base_calc,
             #trainer_calc,
             n_ensembles=num_workers,
             n_cores='max'
             )

structure_optim = Relaxation(images[0],BFGS,fmax=0.05,steps = 100)

if os.path.exists('dft_calls.db'):
    os.remove('dft_calls.db')
structure_optim.run(onlinecalc,filename="relax_oal")



