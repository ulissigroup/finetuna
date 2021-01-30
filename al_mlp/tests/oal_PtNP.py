import ase.io
from online_relaxation_test import run_oal
from al_mlp.atomistic_methods import Relaxation
from ase.calculators.emt import EMT
import numpy as np

#Set up parent calculator and image environment
initial_structure = ase.io.read("./relaxation_test_structures/Pt-NP.traj")
initial_structure.set_calculator(EMT())

EMT_structure_optim = Relaxation(initial_structure,BFGS,fmax=0.05,steps = 100)
EMT_structure_optim.run(EMT(), 'PtNP_emt')
run_oal = run_oal(initial_structure)

def oal_PtNP_energy():
    assert(allclose(EMT_structure_optim.get_potential_energy(), test_oal.get_potential_energy()))
