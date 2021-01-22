import ase
import ase.io
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import VelocityVerlet, Langevin, nvtberendsen

import numpy as np


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
        except:
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
    def __init__(self, initial_geometry, optimizer, fmax=0.05, steps=None):
        self.initial_geometry = initial_geometry
        self.optimizer = optimizer
        self.fmax = fmax
        self.steps = steps

    def run(self, calc, filename):
        structure = self.initial_geometry.copy()
        structure.set_calculator(calc)
        dyn = self.optimizer(structure, trajectory="{}.traj".format(filename))
        dyn.run(fmax=self.fmax, steps=self.steps)

    def get_trajectory(self, filename):
        trajectory = ase.io.Trajectory(filename + ".traj")
        return trajectory
