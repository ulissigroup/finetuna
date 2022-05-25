[![ulissigroup](https://circleci.com/gh/ulissigroup/finetuna.svg?style=svg)](https://app.circleci.com/pipelines/github/ulissigroup/finetuna)
## *FINETUNA*: Fine-Tuning Accelerated Molecular Simulations

Implements active learning with pre-trained graph model fine-tuning to accelerate atomistic simulations.

### Installation

Install dependencies:

- Ensure conda is up-to-date: 
    ```
    conda update conda
    ```

- Create the environment,
    - on a CPU machine:
        ```
        conda env create -f env.cpu.yml
        ```

    - on a GPU machine:
    check the instruction [here](https://github.com/Open-Catalyst-Project/ocp#gpu-machines), and
        ```
        conda env create -f env.gpu.yml
        ```
        
- Activate the conda environment
    ```
    conda activate finetuna
    ```
- Install the package:
    ```
    pip install -e .
    ```

- Install VASP Interactive:
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
