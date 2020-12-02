[![ulissigroup](https://circleci.com/gh/ulissigroup/al_mlp.svg?style=svg)](https://app.circleci.com/pipelines/github/ulissigroup/al_mlp)
## *al_mlp*: Active Learning for Machine Learning Potentials

TODO

### Installation

1. ```pip install git+https://github.com/ulissigroup/al_mlp.git```


### Usage
#### Configs [wip]
```
learner_params = {
    "atomistic_method": Relaxation(          #
        initial_geometry=slab.copy(), 
        optimizer=BFGS, 
        fmax=0.01, 
        steps=100
         )
    "max_iterations": int,                   #
    "samples_to_retrain": int,               #
    "filename": str,                         #
    "file_dir": str,                         #
    "use_dask": bool,                        #
}

```

#### Specify Calculators and Training Data
```
trainer = object
#An isntance of a trainer that has a train and predict method.

training_data = list
#A list of ase.Atoms objects that have attached calculators.
 

parent_calc = ase Calculator object
#Calculator used for querying training data.

base_calc = ase Calculator object
#Calculator used to calculate delta data for training.
```
#### Run Learner
```
learner = OfflineActiveLearner(learner_params, trainer, images, parent_calc, base_calc)
learner.learn()

```

