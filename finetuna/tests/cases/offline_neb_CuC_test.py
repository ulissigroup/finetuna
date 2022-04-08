from finetuna.tests.setup.offline_neb_setup import offline_neb, construct_geometries
from finetuna.calcs import CounterCalc
from ase.io import read
from finetuna.atomistic_methods import NEBcalc
import ase
from ase.calculators.emt import EMT
import numpy as np

import unittest


class offline_NEB(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.emt_counter = CounterCalc(EMT())
        ml2relax = True
        cls.total_neb_images = 5
        intermediate_images = 3
        iter = 7
        initial, final = construct_geometries(
            parent_calc=cls.emt_counter, ml2relax=ml2relax
        )
        images = [initial, final]
        cls.EMT_neb = NEBcalc(images)

        cls.EMT_neb.run(cls.emt_counter, filename="emt_neb")

        initial1, final1 = construct_geometries(parent_calc=EMT(), ml2relax=ml2relax)
        images_neb = [initial1, final1]
        cls.neb_AL_learner = offline_neb(images_neb, EMT(), iter, intermediate_images)
        cls.saddle_pt = int(cls.total_neb_images / 2)
        cls.neb_AL_image = read(
            "example_iter_" + str(iter) + ".traj@" + str(-cls.saddle_pt)
        )
        cls.description = "CuC_NEB"
        return super().setUpClass()

    def get_trajectory(self, filename):
        atom_list = []
        trajectory = ase.io.Trajectory(filename + ".traj")
        for atom in trajectory:
            atom_list.append(atom)
        atom_list = atom_list[-self.total_neb_images :]
        return atom_list

    def test_neb_CuC_energy(self):
        EMT_image = self.EMT_neb.get_trajectory("emt_neb")[-self.saddle_pt]
        neb_AL_image = self.neb_AL_image.copy()
        neb_AL_image.set_calculator(EMT())
        print(EMT_image.get_potential_energy(), neb_AL_image.get_potential_energy())
        assert np.allclose(
            EMT_image.get_potential_energy(),
            neb_AL_image.get_potential_energy(),
            atol=0.05,
        )

    def test_neb_CuC_forces(self):
        neb_AL_image = self.neb_AL_image.copy()
        neb_AL_image.set_calculator(EMT())
        forces = neb_AL_image.get_forces()
        fmax = np.sqrt((forces**2).sum(axis=1).max())
        print(neb_AL_image.get_forces())
        print(fmax)
        assert fmax <= 0.2

    def test_neb_CuC_calls(self):

        print(self.neb_AL_learner.parent_calls)
        print(self.emt_counter.force_calls)
        assert self.neb_AL_learner.parent_calls < 0.5 * self.emt_counter.force_calls
