import os
import copy
from ase.calculators.singlepoint import SinglePointCalculator as sp

def convert_to_singlepoint(images):
    """
    Replaces the attached calculators with singlepoint calculators
    
    Parameters
    ----------
    
    images: list
        List of ase atoms images with attached calculators for forces and energies.
    """
    
    images = copy_images(images)
    singlepoint_images = []
    cwd = os.getcwd()
    for image in images:
        os.makedirs("./temp", exist_ok=True)
        os.chdir("./temp")
        sample_energy = image.get_potential_energy(apply_constraint=False)
        sample_forces = image.get_forces(apply_constraint=False)
        image.set_calculator(
            sp(atoms=image, energy=float(sample_energy), forces=sample_forces)
        )
        singlepoint_images.append(image)
        os.chdir(cwd)
        os.system("rm -rf ./temp")
        
    return singlepoint_images

def compute_with_calc(images, calculator):
    """
    Calculates forces and energies of images with calculator.
    Returned images have singlepoint calculators.
    
    Parameters
    ----------
    
    images: list
        List of ase atoms images to be calculated.
    calc: ase Calculator object
        Calculator used to get forces and energies.
    """
    
    images = copy_images(images)
    for image in images:
        image.set_calculator(copy.deepcopy(calculator))
    return convert_to_singlepoint(images)


def copy_images(images):
    """
    Copies images and returns the new instances.
    The new images also have copied calculators.
    
    Parameters
    ----------
    
    images: list
        List of ase atoms images to be copied.
    """
    new_images = []
    for image in images:
        calc = image.get_calculator()
        new_image = image.copy()
        new_image.set_calculator(copy.deepcopy(calc))
        new_images.append(new_image)
    return new_images
    
    