import unittest

from .online_relaxation_test import run_oal
from al_mlp.atomistic_methods import Relaxation
from al_mlp.utils import CounterCalc
from ase.calculators.emt import EMT
import numpy as np
from ase.cluster.icosahedron import Icosahedron
from ase.optimize import BFGS


class oal_CuNP(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Set up parent calculator and image environment
        initial_structure = Icosahedron("Cu", 2)
        initial_structure.rattle(0.1)
        initial_structure.set_pbc(True)
        initial_structure.set_cell([20, 20, 20])

        EMT_initial_structure = initial_structure.copy()
        cls.emt_counter = CounterCalc(EMT())
        EMT_initial_structure.set_calculator(cls.emt_counter)
        cls.EMT_structure_optim = Relaxation(
            EMT_initial_structure, BFGS, fmax=0.05, steps=30
        )
        cls.EMT_structure_optim.run(cls.emt_counter, "CuNP_emt")

        OAL_initial_structure = initial_structure.copy()
        OAL_initial_structure.set_calculator(EMT())
        OAL_relaxation = Relaxation(
            OAL_initial_structure, BFGS, fmax=0.05, steps=30, maxstep=0.04
        )
        cls.OAL_learner, cls.OAL_structure_optim = run_oal(
            OAL_relaxation, [OAL_initial_structure], "CuNP_oal", EMT()
        )
        cls.EMT_image = cls.EMT_structure_optim.get_trajectory("CuNP_emt")[-1]
        cls.EMT_image.set_calculator(EMT())
        cls.OAL_image = cls.OAL_structure_optim.get_trajectory("CuNP_oal")[-1]
        cls.OAL_image.set_calculator(EMT())
        cls.description = "CuNP"
        return super().setUpClass()

    def test_oal_CuNP_energy(self):
        assert np.allclose(
            self.EMT_image.get_potential_energy(),
            self.OAL_image.get_potential_energy(),
            atol=0.1,
        )

    def test_oal_CuNP_forces(self):
        assert np.allclose(
            self.EMT_image.get_forces(), self.OAL_image.get_forces(), atol=0.05
        )

    def test_oal_CuNP_calls(self):

        # What I want here is the number of EMT calls; I don't think that this is
        # what get_trajectory actually does
        assert self.OAL_learner.parent_calls < 0.5 * self.emt_counter.force_calls
