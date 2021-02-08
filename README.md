[![ulissigroup](https://circleci.com/gh/ulissigroup/al_mlp.svg?style=svg)](https://app.circleci.com/pipelines/github/ulissigroup/al_mlp)
## *al_mlp*: Active Learning for Machine Learning Potentials

Implements active learning with delta machine learning to accelerate atomistic simulations.

### Installation

Install dependencies:

1. Ensure conda is up-to-date: ```conda update conda```

2. Merge the amptorch conda environment with this one and create it:
```
wget https://raw.githubusercontent.com/ulissigroup/amptorch/master/env_cpu.yml -O env_amptorch_cpu.yml
conda-merge env_amptorch_cpu.yml env_cpu.yml > env.yml
conda env create -f env.yml
```

3. Activate the conda environment `conda activate al_mlp` and install the package with `pip install -e .`

### Usage
#### Configs [wip]
```
learner_params = {
    "atomistic_method": Relaxation(          #
        initial_geometry=ase.Atoms object,   # The Atoms object
        optimizer=ase.Optimizer object,      # Optimizer for Relaxation of Starting Image
        fmax=float,                          # Force criteria required to terminate relaxation early
        steps=int                            # Maximum number of steps before relaxation termination
         )
    "max_iterations": int,                   # Maximum number of iterations for the active learning loop
    "samples_to_retrain": int,               # Number of samples to be retrained and added to the training data
    "filename": str,                         # Name of trajectory file generated
    "file_dir": str,                         # Directory where trajectory file will be generated
    "use_dask": bool,                        #
}

```

#### Specify Calculators and Training Data
```
trainer = object                             # An isntance of a trainer that has a train and predict method.

training_data = list                         # A list of ase.Atoms objects that have attached calculators.

parent_calc = ase Calculator object          # Calculator used for querying training data.

base_calc = ase Calculator object<           # Calculator used to calculate delta data for training.
```
#### Run Learner
```
learner = OfflineActiveLearner(learner_params, trainer, training_data, parent_calc, base_calc)
learner.learn()
```
