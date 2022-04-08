import ase.io
from finetuna.tests.test_setup.online_relaxation_test import run_oal
from finetuna.atomistic_methods import Relaxation
from ase.calculators.emt import EMT
from ase.optimize import BFGS
import numpy as np

# Set up parent calculator and image environment
initial_structure = ase.io.read("./relaxation_test_structures/Pt-NP.traj")
initial_structure.set_calculator(EMT())

EMT_structure_optim = Relaxation(initial_structure, BFGS, fmax=0.05, steps=100)
EMT_structure_optim.run(EMT(), "PtNP_emt")
run_oal = run_oal(initial_structure)


def oal_PtNP_energy():
    assert np.allclose(
        EMT_structure_optim.get_potential_energy(), run_oal.get_potential_energy()
    )
