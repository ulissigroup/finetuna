import ase


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
