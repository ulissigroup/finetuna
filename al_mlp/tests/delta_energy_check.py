# Tests if delta calculator is working properly with simple assertion function
from finetuna.calcs import DeltaCalc
from ase.calculators.emt import EMT
import numpy as np
import copy
from ase.build import fcc100, add_adsorbate, molecule
from ase.constraints import FixAtoms
from ase.build import bulk
from ase.utils.eos import EquationOfState
from finetuna.base_calcs.morse import MultiMorse

parent_calculator = EMT()
energies = []
volumes = []
LC = [3.5, 3.55, 3.6, 3.65, 3.7, 3.75]

for a in LC:
    cu_bulk = bulk("Cu", "fcc", a=a)
    calc = EMT()
    cu_bulk.set_calculator(calc)
    e = cu_bulk.get_potential_energy()
    energies.append(e)
    volumes.append(cu_bulk.get_volume())


eos = EquationOfState(volumes, energies)
v0, e0, B = eos.fit()
aref = 3.6
vref = bulk("Cu", "fcc", a=aref).get_volume()
copper_lattice_constant = (v0 / vref) ** (1 / 3) * aref
slab = fcc100("Cu", a=copper_lattice_constant, size=(2, 2, 3))
ads = molecule("C")
add_adsorbate(slab, ads, 2, offset=(1, 1))
cons = FixAtoms(indices=[atom.index for atom in slab if (atom.tag == 3)])
slab.set_constraint(cons)
slab.center(vacuum=13.0, axis=2)
slab.set_pbc(True)
slab.wrap(pbc=[True] * 3)
slab.set_calculator(copy.copy(parent_calculator))
slab.set_initial_magnetic_moments()
images = [slab]
parent_energy = parent_ref = slab.get_potential_energy()


Gs = {
    "default": {
        "G2": {
            "etas": np.logspace(np.log10(0.05), np.log10(5.0), num=4),
            "rs_s": [0],
        },
        "G4": {"etas": [0.005], "zetas": [1.0, 4.0], "gammas": [1.0, -1.0]},
        "cutoff": 6,
    },
}

# create image with base calculator attached
cutoff = Gs["default"]["cutoff"]
base_calc = MultiMorse(images, cutoff, combo="mean")
slab_base = slab.copy()
slab_base.set_calculator(base_calc)
base_energy = base_ref = slab_base.get_potential_energy()

# Add
delta_calc = DeltaCalc([parent_calculator, base_calc], "add", [slab, slab_base])
# Set slab calculator to delta calc and evaluate energy
slab_add = slab.copy()
slab_add.set_calculator(delta_calc)
add_energy = slab_add.get_potential_energy()


# Sub
delta_calc = DeltaCalc([parent_calculator, base_calc], "sub", [slab, slab_base])
# Set slab calculator to delta calc and evaluate energy
slab_sub = slab.copy()
slab_sub.set_calculator(delta_calc)
sub_energy = slab_sub.get_potential_energy()


def test_delta_sub():
    assert sub_energy == (
        (parent_energy - base_energy) + (-parent_ref + base_ref)
    ), "Energies don't match!"


def test_delta_add():
    assert (
        np.abs(add_energy - ((base_energy + parent_energy) + (parent_ref - base_ref)))
        < 1e-5
    ), "Energies don't match!"
