import ase
from ase.neb import SingleCalculatorNEB, NEBTools
from ase.optimize import BFGS
from ase.io import read
from ase.build import fcc100, add_adsorbate
from ase.constraints import FixAtoms
import copy
from ase.calculators.calculator import Calculator


class CounterCalc(Calculator):
    implemented_properties = ["energy", "forces", "uncertainty"]
    """
    Parameters
    --------------
        calc: object. Parent calculator to track force calls"""

    def __init__(self, calc, **kwargs):
        super().__init__()
        self.calc = copy.deepcopy(calc)
        self.force_calls = 0

    def calculate(self, atoms, properties, system_changes):
        super().calculate(atoms, properties, system_changes)
        calc = self.calc
        self.results["energy"] = calc.get_potential_energy(atoms)
        self.results["forces"] = calc.get_forces(atoms)
        self.force_calls += 1


class NEBcalc:
    def __init__(self, starting_images, ml2relax=True, intermediate_samples=3):
        """
        Computes a NEB given an initial and final image.

        Parameters
        ----------
        starting_images: list. Initial and final images to be used for the NEB.

        ml2relax: boolean. True to use ML to relax the initial and final structure guesses.
        False if initial and final structures were relaxed beforehand.

        intermediate_samples: int. Number of intermediate samples to be used in constructing the NEB"""

        self.starting_images = copy.deepcopy(starting_images)
        self.ml2relax = ml2relax
        self.intermediate_samples = intermediate_samples

    def run(self, calc, filename):
        """
        Runs NEB calculations.
        Parameters
        ----------
        calc: object. Calculator to be used to run method.
        filename: str. Label to save generated trajectory files."""

        initial = self.starting_images[0].copy()
        final = self.starting_images[-1].copy()
        if self.ml2relax:
            # Relax initial and final images
            ml_initial = initial
            ml_initial.set_calculator(calc)
            ml_final = final
            ml_final.set_calculator(calc)
            print("BUILDING INITIAL")
            qn = BFGS(
                ml_initial, trajectory="initial.traj", logfile="initial_relax_log.txt"
            )
            qn.run(fmax=0.01, steps=100)
            print("BUILDING FINAL")
            qn = BFGS(ml_final, trajectory="final.traj", logfile="final_relax_log.txt")
            qn.run(fmax=0.01, steps=100)
            initial = ml_initial.copy()
            final = ml_final.copy()

        initial.set_calculator(calc)
        final.set_calculator(calc)

        images = [initial]
        for i in range(self.intermediate_samples):
            image = initial.copy()
            image.set_calculator(calc)
            images.append(image)
        images.append(final)

        print("NEB BEING BUILT")
        neb = SingleCalculatorNEB(images)
        neb.interpolate()
        print("NEB BEING OPTIMISED")
        opti = BFGS(neb, trajectory=filename + ".traj", logfile="al_neb_log.txt")
        opti.run(fmax=0.01, steps=100)
        print("NEB DONE")

    def get_trajectory(self, filename):
        atom_list = []
        trajectory = ase.io.Trajectory(filename + ".traj")
        for atom in trajectory:
            atom_list.append(atom)
        s = self.intermediate_samples + 2
        atom_list = atom_list[-s:]
        return atom_list


def construct_geometries(parent_calc, ml2relax):
    counter_calc = parent_calc
    # Initial structure guess
    initial_slab = fcc100("Cu", size=(2, 2, 3))
    add_adsorbate(initial_slab, "C", 1.7, "hollow")
    initial_slab.center(axis=2, vacuum=4.0)
    mask = [atom.tag > 1 for atom in initial_slab]
    initial_slab.set_constraint(FixAtoms(mask=mask))
    initial_slab.set_pbc(True)
    initial_slab.wrap(pbc=[True] * 3)
    initial_slab.set_calculator(counter_calc)

    # Final structure guess
    final_slab = initial_slab.copy()
    final_slab[-1].x += final_slab.get_cell()[0, 0] / 3
    final_slab.set_calculator(counter_calc)
    if not ml2relax:
        print("BUILDING INITIAL")
        qn = BFGS(
            initial_slab, trajectory="initial.traj", logfile="initial_relax_log.txt"
        )
        qn.run(fmax=0.01, steps=100)
        print("BUILDING FINAL")
        qn = BFGS(final_slab, trajectory="final.traj", logfile="final_relax_log.txt")
        qn.run(fmax=0.01, steps=100)
        initial_slab = read("initial.traj", "-1")
        final_slab = read("final.traj", "-1")
        # If there is already a pre-existing initial and final relaxed parent state we can read that to use as a starting point
        # initial_slab = read("/content/parent_initial.traj")
        # final_slab = read("/content/parent_final.traj")
    else:
        initial_slab = initial_slab
        final_slab = final_slab

    # initial_force_calls = counter_calc.force_calls
    return initial_slab, final_slab  # , initial_force_calls
