import unittest

import ase.io
from al_mlp.tests.test_setup.online_relaxation_test import run_oal
from al_mlp.atomistic_methods import Relaxation
from al_mlp.calcs import CounterCalc
from ase.calculators.emt import EMT
from ase.optimize import BFGS
import numpy as np


class oal_PtNP(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Set up parent calculator and image environment
        initial_structure = ase.io.read("./relaxation_test_structures/Pt-NP.traj")
        initial_structure.set_calculator(EMT())

        # Run relaxation with the parent calc
        EMT_initial_structure = initial_structure.copy()
        cls.emt_counter = CounterCalc(EMT())
        EMT_initial_structure.set_calculator(cls.emt_counter)
        cls.EMT_structure_optim = Relaxation(
            EMT_initial_structure, BFGS, fmax=0.05, steps=100
        )
        cls.EMT_structure_optim.run(cls.emt_counter, "PtNP_emt")

        # Run relaxation with active learning
        OAL_initial_structure = initial_structure.copy()
        OAL_initial_structure.set_calculator(EMT())
        OAL_relaxation = Relaxation(
            OAL_initial_structure, BFGS, fmax=0.05, steps=30, maxstep=0.04
        )
        cls.OAL_learner, cls.OAL_structure_optim = run_oal(
            OAL_relaxation, [OAL_initial_structure], "PtNP_oal", EMT()
        )

        # Retain images of the final structure from both relaxations
        cls.EMT_image = cls.EMT_structure_optim.get_trajectory("PtNP_emt")[-1]
        cls.EMT_image.set_calculator(EMT())
        cls.OAL_image = cls.OAL_structure_optim.get_trajectory("PtNP_oal")[-1]
        cls.OAL_image.set_calculator(EMT())
        cls.description = "PtNP"
        return super().setUpClass()

    def test_oal_PtNP_energy(self):
        assert np.allclose(
            self.EMT_image.get_potential_energy(),
            self.OAL_image.get_potential_energy(),
            atol=0.1,
        )

    def test_oal_PtNP_forces(self):
        assert np.allclose(
            self.EMT_image.get_forces(), self.OAL_image.get_forces(), atol=0.05
        )

    def test_oal_PtNP_calls(self):
        assert self.OAL_learner.parent_calls < 0.5 * self.emt_counter.force_calls
