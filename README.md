[![ulissigroup](https://circleci.com/gh/ulissigroup/finetuna.svg?style=svg)](https://app.circleci.com/pipelines/github/ulissigroup/finetuna)
## *FINETUNA*: Fine-Tuning Accelerated Molecular Simulations

Implements active learning with pre-trained graph model fine-tuning to accelerate atomistic simulations.

### Installation

Install dependencies:

1. Ensure conda is up-to-date: 
```
conda update conda
```

2. Merge the amptorch conda environment with this one and create it:
```
conda env create -f env.cpu.yml.yml
```
3. Activate the conda environment
```
conda activate finetuna
```
4. Install the package:
```
pip install -e .
```

5. Install VASP Interactive:
```
pip install git+https://github.com/ulissigroup/vasp-interactive.git
```

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
