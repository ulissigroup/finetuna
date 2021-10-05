import unittest

from al_mlp.tests.setup.online_relaxation_test import run_online_al
from al_mlp.atomistic_methods import Relaxation
from al_mlp.calcs import CounterCalc
from ase.calculators.emt import EMT
import numpy as np
from ase.cluster.icosahedron import Icosahedron
from ase.optimize import BFGS

FORCE_THRESHOLD = 0.05
ENERGY_THRESHOLD = 0.01


class online_CuNP(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Set up parent calculator and image environment
        initial_structure = Icosahedron("Cu", 2)
        initial_structure.rattle(0.1)
        initial_structure.set_pbc(True)
        initial_structure.set_cell([20, 20, 20])

        # Run relaxation with the parent calc
        EMT_initial_structure = initial_structure.copy()
        cls.emt_counter = CounterCalc(EMT())
        EMT_initial_structure.set_calculator(cls.emt_counter)
        cls.EMT_structure_optim = Relaxation(
            EMT_initial_structure, BFGS, fmax=FORCE_THRESHOLD, steps=30
        )
        cls.EMT_structure_optim.run(cls.emt_counter, "CuNP_emt")

        # Run relaxation with active learning
        OAL_initial_structure = initial_structure.copy()
        OAL_initial_structure.set_calculator(EMT())
        OAL_relaxation = Relaxation(
            OAL_initial_structure, BFGS, fmax=0.05, steps=100, maxstep=0.04
        )
        cls.OAL_learner, cls.OAL_structure_optim = run_online_al(
            OAL_relaxation,
            [],
            ["Cu"],
            "CuNP_oal",
            CounterCalc(EMT()),
        )

        # Retain images of the final structure from both relaxations
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
            atol=ENERGY_THRESHOLD,
        ), str(
            "Learner energy inconsistent:\n"
            + str(self.EMT_image.get_potential_energy())
            + "or Parent energy inconsistent:\n"
            + str(self.OAL_image.get_potential_energy())
            + "\nwith Energy Threshold: "
            + str(ENERGY_THRESHOLD)
            + "\nafter "
            + str(self.OAL_learner.curr_step)
            + " steps"
        )

    def test_oal_CuNP_forces(self):
        forces = self.OAL_image.get_forces()
        fmax = np.sqrt((forces ** 2).sum(axis=1).max())

        assert fmax <= FORCE_THRESHOLD, str(
            "Learner forces inconsistent:\n"
            + str(fmax)
            + "\nwith Force Threshold: "
            + str(FORCE_THRESHOLD)
            + "\nafter "
            + str(self.OAL_learner.curr_step)
            + " steps"
        )

    def test_oal_CuNP_calls(self):

        print("OAL calls: %d" % self.OAL_learner.parent_calls)
        print("EMT calls: %d" % self.emt_counter.force_calls)

        assert self.OAL_learner.parent_calls <= 0.5 * self.emt_counter.force_calls, str(
            "total calls: "
            + str(self.OAL_learner.parent_calls)
            + " not less than: "
            + str(0.5 * self.emt_counter.force_calls)
            + "\nafter "
            + str(self.OAL_learner.curr_step)
            + " steps"
        )

    def test_oal_CuNP_calls_consistent(self):
        assert (
            self.OAL_learner.parent_calls == self.OAL_learner.parent_calc.force_calls
        ), str(
            "number of parent calls tracked by learner: "
            + str(self.OAL_learner.parent_calls)
            + ", is not equal to number of calls tracked by parent call counter: "
            + str(self.OAL_learner.parent_calc.force_calls)
        )
