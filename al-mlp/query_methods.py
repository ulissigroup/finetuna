import os
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator as sp
import random
import copy
import warnings

"""
All query methods should have the following arguments
    Paramaters
    ----------
    images: List. Current training images.

    sample_candidates: List. Potential candidates for query as collected
    from the generating function.

    samples_to_retrain: Integer. Number of samples to be randomly selected
    for query and added to the training set.

    parent_calc: Object. Parent calculator used to query energy and force
    calculations.
"""


def random_query(_, sample_candidates, samples_to_retrain, parent_calc):
    """
    Randomly selects data points from a list of potential candidates to
    query and add to the existing training set.
    """
    if len(sample_candidates) <= samples_to_retrain:
        warnings.warn(
            "# of samples exceeds # of available candidates! Defaulting to all available candidates",
            stacklevel=2,
        )
        samples_to_retrain = len(sample_candidates) - 1
    query_idx = random.sample(range(1, len(sample_candidates)), samples_to_retrain)
    images_to_query = [sample_candidates[idx] for idx in query_idx]
    queried_images = compute_query(images_to_query, parent_calc)
    return queried_images


def max_uncertainty(_, sample_candidates, samples_to_retrain, parent_calc):
    """Selects points with the largest uncertainty"""
    if len(sample_candidates) < samples_to_retrain:
        warnings.warn(
            "# of samples exceeds # of available candidates! Defaulting to all available candidates",
            stacklevel=2,
        )
        samples_to_retrain = len(sample_candidates) - 1
    uncertainty = np.array(
        [atoms.info["uncertainty"][0] for atoms in sample_candidates]
    )
    query_idx = np.argpartition(uncertainty, -1 * samples_to_retrain)[
        -1 * samples_to_retrain :
    ]
    images_to_query = [sample_candidates[idx] for idx in query_idx]
    queried_images = compute_query(images_to_query, parent_calc)
    return queried_images


def final_query(_, sample_candidates, samples_to_retrain, parent_calc):
    if len(sample_candidates) <= samples_to_retrain:
        warnings.warn(
            "# of samples exceeds # of available candidates! Defaulting to all available candidates",
            stacklevel=2,
        )
        samples_to_retrain = len(sample_candidates) - 1
    query_idx = random.sample(range(1, len(sample_candidates)-1), samples_to_retrain)
    query_idx.append(-1)
    images_to_query = [sample_candidates[idx] for idx in query_idx]
    queried_images = compute_query(images_to_query, parent_calc)
    return queried_images


def compute_query(images_to_calculate, parent_calc):
    queried_images = []
    cwd = os.getcwd()
    for image in images_to_calculate:
        os.makedirs("./temp", exist_ok=True)
        os.chdir("./temp")
        image.set_calculator(copy.copy(parent_calc))
        sample_energy = image.get_potential_energy(apply_constraint=False)
        sample_forces = image.get_forces(apply_constraint=False)
        image.set_calculator(
            sp(atoms=image, energy=sample_energy, forces=sample_forces)
        )
        queried_images.append(image)
        os.chdir(cwd)
        os.system("rm -rf ./temp")
    return queried_images


def termination_criteria(termination_args, method="iter"):
    """Criteria for AL termination
    Parameters
    ----------
    method: str
        Method for termination of active learning loop.

    'iter': Terminates after specified number of iterations.
        args: (current iteration, # of iterations)
    """
    terminate = False

    if method == "iter":
        current_i = termination_args["current_i"]
        total_i = termination_args["total_i"]
        if current_i > total_i:
            terminate = True
    if method == "final":
        ml_dft_image = termination_args["queried"][-1]
        f_tol = termination_args["force_tol"]

        max_f = np.max(np.abs(ml_dft_image.get_forces()))
        terminate = max_f < f_tol
    return terminate
