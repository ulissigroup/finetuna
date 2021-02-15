from offline_neb_Cu_C_utils import construct_geometries
from al_mlp.calcs import CounterCalc
from al_mlp.atomistic_methods import NEBcalc
from offline_neb_learner import offline_neb
import ase
from ase.calculators.emt import EMT
import numpy as np


def get_trajectory(filename):
    atom_list = []
    trajectory = ase.io.Trajectory(filename + ".traj")
    for atom in trajectory:
        atom_list.append(atom)
    s = 5
    atom_list = atom_list[-s:]
    return atom_list


emt_counter = CounterCalc(EMT())
ml2relax = True
total_neb_images = 5
intermediate_images = 3
iter = 6
initial, final = construct_geometries(parent_calc=emt_counter, ml2relax=ml2relax)
images = [initial, final]
EMT_neb = NEBcalc(images)

EMT_neb.run(emt_counter, filename="emt_neb")

initial1, final1 = construct_geometries(parent_calc=EMT(), ml2relax=ml2relax)
images_neb = [initial1, final1]
neb_AL_learner = offline_neb(EMT(), iter, intermediate_images)
saddle_pt = int(total_neb_images / 2)
neb_AL_image = get_trajectory("example_iter_" + str(iter))[-saddle_pt]


def neb_CuC_energy():
    EMT_image = EMT_neb.get_trajectory("emt_neb")[-3]
    EMT_image.set_calculator(EMT())
    neb_AL_image = get_trajectory("example_iter_" + str(iter))[-saddle_pt]
    neb_AL_image.set_calculator(EMT())
    assert np.allclose(
        EMT_image.get_potential_energy(),
        neb_AL_image.get_potential_energy(),
        atol=0.1,
    )


def neb_CuC_forces():
    EMT_image = EMT_neb.get_trajectory("emt_neb")[-3]
    EMT_image.set_calculator(EMT())
    neb_AL_image = get_trajectory("example_iter_" + str(iter))[-saddle_pt]
    neb_AL_image.set_calculator(EMT())

    assert np.max(np.abs(neb_AL_image.get_forces())) <= 0.05


def neb_CuC_calls():

    # What I want here is the number of EMT calls; I don't think that this is
    # what get_trajectory actually does
    assert neb_AL_learner.parent_calls < 0.5 * emt_counter.force_calls


neb_CuC_energy()
neb_CuC_forces()
neb_CuC_calls()
