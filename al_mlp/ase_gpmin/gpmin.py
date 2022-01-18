import numpy as np
import warnings

from scipy.optimize import minimize
from ase.parallel import world
from ase.io.jsonio import write_json
from ase.optimize.optimize import Optimizer
from al_mlp.ase_gpmin.gp import GaussianProcess
from ase.optimize.gpmin.kernel import SquaredExponential
from ase.optimize.gpmin.prior import ConstantPrior
from al_mlp.ase_gpmin.prior import OCPPrior


class GPMin(Optimizer, GaussianProcess):
    def __init__(self, atoms, restart=None, logfile='-', trajectory=None,
                 prior=None, kernel=None, master=None, noise=None, weight=None,
                 scale=None, force_consistent=None, batch_size=None,
                 bounds=None, update_prior_strategy="maximum",
                 update_hyperparams=False):
        """Optimize atomic positions using GPMin algorithm, which uses both
        potential energies and forces information to build a PES via Gaussian
        Process (GP) regression and then minimizes it.

        Default behaviour:
        --------------------
        The default values of the scale, noise, weight, batch_size and bounds
        parameters depend on the value of update_hyperparams. In order to get
        the default value of any of them, they should be set up to None.
        Default values are:

        update_hyperparams = True
            scale : 0.3
            noise : 0.004
            weight: 2.
            bounds: 0.1
            batch_size: 1

        update_hyperparams = False
            scale : 0.4
            noise : 0.005
            weight: 1.
            bounds: irrelevant
            batch_size: irrelevant

        Parameters:
        ------------------

        atoms: Atoms object
            The Atoms object to relax.

        restart: string
            JSON file used to store the training set. If set, file with
            such a name will be searched and the data in the file incorporated
            to the new training set, if the file exists.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout

        trajectory: string
            File used to store trajectory of atomic movement.

        master: boolean
            Defaults to None, which causes only rank 0 to save files. If
            set to True, this rank will save files.

        force_consistent: boolean or None
            Use force-consistent energy calls (as opposed to the energy
            extrapolated to 0 K). By default (force_consistent=None) uses
            force-consistent energies if available in the calculator, but
            falls back to force_consistent=False if not.

        prior: Prior object or None
            Prior for the GP regression of the PES surface
            See ase.optimize.gpmin.prior
            If *prior* is None, then it is set as the
            ConstantPrior with the constant being updated
            using the update_prior_strategy specified as a parameter

        kernel: Kernel object or None
            Kernel for the GP regression of the PES surface
            See ase.optimize.gpmin.kernel
            If *kernel* is None the SquaredExponential kernel is used.
            Note: It needs to be a kernel with derivatives!!!!!

        noise: float
            Regularization parameter for the Gaussian Process Regression.

        weight: float
            Prefactor of the Squared Exponential kernel.
            If *update_hyperparams* is False, changing this parameter
            has no effect on the dynamics of the algorithm.

        update_prior_strategy: string
            Strategy to update the constant from the ConstantPrior
            when more data is collected. It does only work when
            Prior = None

            options:
                'maximum': update the prior to the maximum sampled energy
                'init' : fix the prior to the initial energy
                'average': use the average of sampled energies as prior

        scale: float
            scale of the Squared Exponential Kernel

        update_hyperparams: boolean
            Update the scale of the Squared exponential kernel
            every batch_size-th iteration by maximizing the
            marginal likelihood.

        batch_size: int
            Number of new points in the sample before updating
            the hyperparameters.
            Only relevant if the optimizer is executed in update_hyperparams
            mode: (update_hyperparams = True)

        bounds: float, 0<bounds<1
            Set bounds to the optimization of the hyperparameters.
            Let t be a hyperparameter. Then it is optimized under the
            constraint (1-bound)*t_0 <= t <= (1+bound)*t_0
            where t_0 is the value of the hyperparameter in the previous
            step.
            If bounds is False, no constraints are set in the optimization of
            the hyperparameters.

        .. warning:: The memory of the optimizer scales as O(n²N²) where
                     N is the number of atoms and n the number of steps.
                     If the number of atoms is sufficiently high, this
                     may cause a memory issue.
                     This class prints a warning if the user tries to
                     run GPMin with more than 100 atoms in the unit cell.
        """
        # Warn the user if the number of atoms is very large
        if len(atoms) > 100:
            warning = ('Possible Memory Issue. There are more than '
                       '100 atoms in the unit cell. The memory '
                       'of the process will increase with the number '
                       'of steps, potentially causing a memory issue. '
                       'Consider using a different optimizer.')

            warnings.warn(warning)

        # Give it default hyperparameters
        if update_hyperparams:       # Updated GPMin
            if scale is None:
                scale = 0.3
            if noise is None:
                noise = 0.004
            if weight is None:
                weight = 2.
            if bounds is None:
                self.eps = 0.1
            elif bounds is False:
                self.eps = None
            else:
                self.eps = bounds
            if batch_size is None:
                self.nbatch = 1
            else:
                self.nbatch = batch_size
        else:                        # GPMin without updates
            if scale is None:
                scale = 0.4
            if noise is None:
                noise = 0.001
            if weight is None:
                weight = 1.
            if bounds is not None:
                warning = ('The parameter bounds is of no use '
                           'if update_hyperparams is False. '
                           'The value provided by the user '
                           'is being ignored.')
                warnings.warn(warning, UserWarning)
            if batch_size is not None:
                warning = ('The parameter batch_size is of no use '
                           'if update_hyperparams is False. '
                           'The value provided by the user '
                           'is being ignored.')
                warnings.warn(warning, UserWarning)

            # Set the variables to something anyways
            self.eps = False
            self.nbatch = None

        self.strategy = update_prior_strategy
        self.update_hp = update_hyperparams
        self.function_calls = 1
        self.force_calls = 0
        self.x_list = []      # Training set features
        self.y_list = []      # Training set targets

        Optimizer.__init__(self, atoms, restart, logfile,
                           trajectory, master, force_consistent)
        if prior is None:
            self.update_prior = True
            prior = ConstantPrior(constant=None)
        else:
            self.update_prior = False

        if kernel is None:
            kernel = SquaredExponential()
        GaussianProcess.__init__(self, prior, kernel)
        self.set_hyperparams(np.array([weight, scale, noise]))

    def acquisition(self, r):
        e = self.predict(r)
        return e[0], e[1:]

    def update(self, r, e, f):
        """Update the PES

        Update the training set, the prior and the hyperparameters.
        Finally, train the model
        """
        # update the training set
        self.x_list.append(r)
        f = f.reshape(-1)
        y = np.append(np.array(e).reshape(-1), -f)
        self.y_list.append(y)

        # Set/update the constant for the prior
        if self.update_prior:
            if self.strategy == 'average':
                av_e = np.mean(np.array(self.y_list)[:, 0])
                self.prior.set_constant(av_e)
            elif self.strategy == 'maximum':
                max_e = np.max(np.array(self.y_list)[:, 0])
                self.prior.set_constant(max_e)
            elif self.strategy == 'init':
                self.prior.set_constant(e)
                self.update_prior = False

        if isinstance(self.prior, OCPPrior):
            self.prior.parent_initial_energy = self.parent_initial_energy

        # update hyperparams
        if (self.update_hp and self.function_calls % self.nbatch == 0 and
           self.function_calls != 0):
            self.fit_to_batch()

        # build the model
        self.train(np.array(self.x_list), np.array(self.y_list))

    def relax_model(self, r0):
        result = minimize(self.acquisition, r0, method='L-BFGS-B', jac = True,  options={'disp': True})
        if result.success:
            return result.x
        else:
            self.dump()
            raise RuntimeError("The minimization of the acquisition function "
                               "has not converged")

    def fit_to_batch(self):
        """Fit hyperparameters keeping the ratio noise/weight fixed"""
        ratio = self.noise/self.kernel.weight
        self.fit_hyperparameters(np.array(self.x_list),
                                 np.array(self.y_list), eps=self.eps)
        self.noise = ratio*self.kernel.weight

    def step(self, f=None):
        atoms = self.atoms
        if f is None:
            f = atoms.get_forces()

        fc = self.force_consistent
        r0 = atoms.get_positions().reshape(-1)
        e0 = atoms.get_potential_energy(force_consistent=fc)
        if self.nsteps == 0:
            self.parent_initial_energy = e0
        self.update(r0, e0, f)

        r1 = self.relax_model(r0)
        self.atoms.set_positions(r1.reshape(-1, 3))
        e1 = self.atoms.get_potential_energy(force_consistent=fc)
        f1 = self.atoms.get_forces()
        self.function_calls += 1
        self.force_calls += 1
        count = 0
        while e1 >= e0:
            self.update(r1, e1, f1)
            r1 = self.relax_model(r0)

            self.atoms.set_positions(r1.reshape(-1, 3))
            e1 = self.atoms.get_potential_energy(force_consistent=fc)
            f1 = self.atoms.get_forces()
            self.function_calls += 1
            self.force_calls += 1
            if self.converged(f1):
                break

            count += 1
            if count == 30:
                raise RuntimeError("A descent model could not be built")
        self.dump()

    def dump(self):
        """Save the training set"""
        if world.rank == 0 and self.restart is not None:
            with open(self.restart, 'wb') as fd:
                write_json(fd, (self.x_list, self.y_list))

    def read(self):
        self.x_list, self.y_list = self.load()
