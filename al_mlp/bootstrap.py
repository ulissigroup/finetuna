import random
from ase import Atoms


def bootstrap_ensemble(
    parent_dataset, resampled_set=None, new_data=None, n_ensembles=1
):
    if len(parent_dataset) == 1 and new_data is None:
        ensemble_sets = [parent_dataset.copy() for i in range(n_ensembles)]
        return ensemble_sets, parent_dataset
    ensemble_sets = []
    if new_data and resampled_set:
        n_ensembles = len(resampled_set)
        parent_dataset.append(new_data)
        for i in range(n_ensembles):
            sample = random.sample(parent_dataset, 1)[0]
            resampled_set[i].append(sample)
            for k in range(len(resampled_set[i])):
                p = random.random()
                if p < 1 / len(resampled_set[i]):
                    resampled_set[i][k] = new_data
            ensemble_sets.append(resampled_set[i])
    else:
        for i in range(n_ensembles):
            ensemble_sets.append(random.choices(parent_dataset, k=len(parent_dataset)))
    return ensemble_sets, parent_dataset

def non_bootstrap_ensemble(parent_dataset, new_data=None, n_ensembles=1):
    if new_data:
        if isinstance(new_data, list):
            for datum in new_data:
                parent_dataset.append(datum)
        elif isinstance(new_data,Atoms):
            parent_dataset.append(new_data)
        else:
            print("new_data is not an Atoms object or list of Atoms, non_bootstrap_ensemble() failed!")
    for i in range(n_ensembles):
        ensemble_sets = [parent_dataset.copy() for i in range(n_ensembles)]
    return ensemble_sets, parent_dataset

