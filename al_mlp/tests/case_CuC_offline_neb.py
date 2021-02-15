from .test_setup.offline_neb_Cu_C_utils import construct_geometries
from .test_setup.offline_neb_test import offline_neb
from al_mlp.calcs import CounterCalc
from al_mlp.atomistic_methods import NEBcalc
import ase
from ase.calculators.emt import EMT
import numpy as np

import unittest

def get_trajectory(filename):
    atom_list = []
    trajectory = ase.io.Trajectory(filename + ".traj")
    for atom in trajectory:
        atom_list.append(atom)
    s = 5
    atom_list = atom_list[-s:]
    return atom_list


class offline_NEB(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.emt_counter = CounterCalc(EMT())
        ml2relax = True
        total_neb_images = 5
        intermediate_images = 3
        iter = 6
        initial, final = construct_geometries(parent_calc=cls.emt_counter, ml2relax=ml2relax)
        images = [initial, final]
        cls.EMT_neb = NEBcalc(images)
        
        cls.EMT_neb.run(cls.emt_counter, filename="emt_neb")
        
        initial1, final1 = construct_geometries(parent_calc=EMT(), ml2relax=ml2relax)
        images_neb = [initial1, final1]
        cls.neb_AL_learner = offline_neb(images_neb,EMT(), iter, intermediate_images)
        cls.saddle_pt = int(total_neb_images / 2)
        cls.neb_AL_image = get_trajectory("example_iter_" + str(iter))[-cls.saddle_pt]
        cls.description = "CuC_NEB"
        return super().setUpClass()
    
    def test_neb_CuC_energy(self):
        EMT_image = self.EMT_neb.get_trajectory("emt_neb")[-3]
        EMT_image.set_calculator(EMT())
        neb_AL_image = self.neb_AL_image.copy()
        neb_AL_image.set_calculator(EMT())
        assert np.allclose(
            EMT_image.get_potential_energy(),
            neb_AL_image.get_potential_energy(),
            atol=0.1,
        )
    
    
    def test_neb_CuC_forces(self):
        EMT_image = self.EMT_neb.get_trajectory("emt_neb")[-3]
        EMT_image.set_calculator(EMT())
        neb_AL_image = self.neb_AL_image.copy()
        neb_AL_image.set_calculator(EMT())
    
        assert np.max(np.abs(neb_AL_image.get_forces())) <= 0.05
    
    
    def test_neb_CuC_calls(self):
    
        # What I want here is the number of EMT calls; I don't think that this is
        # what get_trajectory actually does
        assert self.neb_AL_learner.parent_calls < 0.5 * self.emt_counter.force_calls

