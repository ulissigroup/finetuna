import random


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
