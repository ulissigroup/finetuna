import ase
import ase.io
from ase.neb import SingleCalculatorNEB
from ase.optimize import BFGS
import copy
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import VelocityVerlet, Langevin, nvtberendsen
import numpy as np


class NEBcalc:
    def __init__(self, starting_images, intermediate_samples=3):
        """
        Computes a NEB given an initial and final image.

        Parameters
        ----------
        starting_images: list. Initial and final images to be used for the NEB.

        intermediate_samples: int. Number of intermediate samples to be used in constructing the NEB"""

        self.starting_images = copy.deepcopy(starting_images)
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


class MDsimulate:
    def __init__(self, thermo_ensemble, dt, temp, count, initial_geometry=None):
        """
        Parameters
        ----------
        ensemble: "NVE", "nvtberendsen", "langevin"
        dt: md time step (fs)
        temp: temperature (K)
        initial_slab: initial geometry to use, if None - will be generated
        """
        self.ensemble = thermo_ensemble
        self.dt = dt
        self.temp = temp
        self.count = count
        if initial_geometry is None:
            raise Exception("Initial structure not provided!")
        else:
            self.starting_geometry = initial_geometry

    def run(self, calc, filename):
        slab = self.starting_geometry.copy()
        slab.set_calculator(calc)
        np.random.seed(1)
        MaxwellBoltzmannDistribution(slab, self.temp * units.kB)
        if self.ensemble == "NVE":
            dyn = VelocityVerlet(slab, self.dt * units.fs)
        elif self.ensemble == "nvtberendsen":
            dyn = nvtberendsen.NVTBerendsen(
                slab, self.dt * units.fs, self.temp, taut=300 * units.fs
            )
        elif self.ensemble == "langevin":
            dyn = Langevin(slab, self.dt * units.fs, self.temp * units.kB, 0.002)
        traj = ase.io.Trajectory(
            filename + ".traj", "w", slab, properties=["energy", "forces"]
        )
        dyn.attach(traj.write, interval=1)
        try:
            fixed_atoms = len(slab.constraints[0].get_indices())
        except Exception:
            fixed_atoms = 0
            pass

        def printenergy(a=slab):
            """Function to print( the potential, kinetic, and total energy)"""
            epot = a.get_potential_energy() / len(a)
            ekin = a.get_kinetic_energy() / (len(a) - fixed_atoms)
            print(
                "Energy per atom: Epot = %.3feV Ekin = %.3feV (T=%3.0fK) "
                "Etot = %.3feV" % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin)
            )

        if printenergy:
            dyn.attach(printenergy, interval=10)
        dyn.run(self.count)

    def get_trajectory(self, filename):
        trajectory = ase.io.Trajectory(filename + ".traj")
        return trajectory


class Relaxation:
    def __init__(
        self, initial_geometry, optimizer, fmax=0.05, steps=None, maxstep=None
    ):
        self.initial_geometry = initial_geometry
        self.optimizer = optimizer
        self.fmax = fmax
        self.steps = steps
        self.maxstep = maxstep

    def run(
        self,
        calc,
        filename,
        replay_traj=False,
        max_parent_calls=None,
        check_final=False,
        online_ml_fmax = None,
    ):
        structure = self.initial_geometry.copy()
        structure.set_calculator(calc)
        if self.maxstep is not None:
            dyn = self.optimizer(
                structure, maxstep=self.maxstep, trajectory="{}.traj".format(filename)
            )
        else:
            dyn = self.optimizer(structure, trajectory="{}.traj".format(filename))

        if replay_traj:
            dyn.attach(replay_trajectory, 1, calc, dyn)

        if max_parent_calls is not None:
            dyn.attach(max_parent_observer, 1, calc, dyn, max_parent_calls)

        if check_final:
            dyn.attach(check_final_point, 1, calc, dyn)

        if online_ml_fmax is not None and online_ml_fmax != self.fmax:
            dyn.parent_fmax = self.fmax
            dyn.ml_fmax = online_ml_fmax
            dyn.attach(set_online_ml_fmax, 1, calc, dyn)

        dyn.run(fmax=self.fmax, steps=self.steps)

    def get_trajectory(self, filename):
        trajectory = ase.io.Trajectory(filename + ".traj")
        return trajectory


def set_online_ml_fmax(calc, optimizer):
    if calc.info.get("check", True):
        optimizer.fmax = optimizer.parent_fmax
    else:
        optimizer.fmax = optimizer.ml_fmax


def check_final_point(calc, optimizer):
    """Check with parent calc when max step is reached"""
    if optimizer.nsteps == optimizer.max_steps - 1:
        calc.check_final_point = True


def max_parent_observer(calc, optimizer, max_parent_calls):
    if calc.parent_calls >= max_parent_calls:
        optimizer.nsteps = optimizer.max_steps


def replay_trajectory(calc, optimizer):
    """Initialize hessian from parent dataset."""
    if calc.info.get("check", False):
        # parent_dataset = calc.parent_dataset
        complete_dataset = calc.complete_dataset
        # check the parent dataset and only use structures that match the final structure
        dataset = []
        if calc.rolling_opt_window is not None:
            if len(complete_dataset) > calc.rolling_opt_window:
                complete_dataset = complete_dataset[-calc.rolling_opt_window :]
        final_atomic_numbers = complete_dataset[-1].get_atomic_numbers()
        for atoms in complete_dataset:
            match_array = atoms.get_atomic_numbers() == final_atomic_numbers
            if type(match_array) is np.ndarray and match_array.all():
                dataset.append(atoms)

        optimizer.H = None
        atoms = dataset[0]
        r0 = atoms.get_positions().ravel()
        f0 = atoms.get_forces(apply_constraint=False).ravel()
        for atoms in dataset:
            if atoms.info.get("check", False):
                r = atoms.get_positions().ravel()
                f = atoms.get_forces(apply_constraint=False).ravel()
            else:
                atoms_copy = atoms.copy()
                atoms_copy.calc = calc.ml_potential
                r = atoms_copy.get_positions().ravel()
                f = atoms_copy.get_forces(apply_constraint=False).ravel()

            optimizer.update(r, f, r0, f0)
            r0 = r
            f0 = f

        optimizer.r0 = r0
        optimizer.f0 = f0
