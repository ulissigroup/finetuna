import numpy as np


class Prior():
    """Base class for all priors for the bayesian optimizer.

    The __init__ method and the prior method are implemented here.
    Each child class should implement its own potential method, that will be
    called by the prior method implemented here.

    When used, the prior should be initialized outside the optimizer and the
    Prior object should be passed as a function to the optimizer.
    """
    def __init__(self):
        """Basic prior implementation."""
        self.initial_point = True
        self.count = 0
        pass

    def prior(self, x):
        """Actual prior function, common to all Priors"""
        if len(x.shape) > 1:
            n = x.shape[0]
            self.initial_point = False
            return np.hstack([self.potential(x[i, :]) for i in range(n)])
        else:
            return self.potential(x)


class ZeroPrior(Prior):
    """ZeroPrior object, consisting on a constant prior with 0eV energy."""
    def __init__(self):
        Prior.__init__(self)

    def potential(self, x):
        return np.zeros(x.shape[0]+1)


class ConstantPrior(Prior):
    """Constant prior, with energy = constant and zero forces

    Parameters:

    constant: energy value for the constant.

    Example:

    >>> from ase.optimize import GPMin
    >>> from ase.optimize.gpmin.prior import ConstantPrior
    >>> op = GPMin(atoms, Prior = ConstantPrior(10)
    """
    def __init__(self, constant):
        self.constant = constant
        Prior.__init__(self)

    def potential(self, x):
        d = x.shape[0]
        output = np.zeros(d+1)
        output[0] = self.constant
        return output

    def set_constant(self, constant):
        self.constant = constant

class CalculatorPrior(Prior):
    """CalculatorPrior object, allows the user to
    use another calculator as prior function instead of the
    default constant.

    Parameters:

    atoms: the Atoms object
    calculator: one of ASE's calculators
    """
    def __init__(self, atoms, calculator):
        Prior.__init__(self)
        self.atoms = atoms.copy()
        self.atoms.calc = calculator

    def potential(self, x):
        self.atoms.set_positions(x.reshape(-1, 3))
        V = self.atoms.get_potential_energy(force_consistent=True)
        gradV = -self.atoms.get_forces().reshape(-1)
        return np.append(np.array(V).reshape(-1), gradV)

class OCPPrior(Prior):
    """OCPPrior object, allows the user to
    use OCP calculator as prior function instead of the
    default constant.

    Parameters:

    atoms: the Atoms object
    calculator: OCP calculator
    """
    def __init__(self, atoms, calculator):
        Prior.__init__(self)
        self.atoms = atoms.copy()
        self.atoms.calc = calculator
        self.parent_initial_energy = None
        self.prior_initial_energy = None

    def potential(self, x):
        self.atoms.set_positions(x.reshape(-1, 3))
        V = self.atoms.get_potential_energy(force_consistent=False)
        if self.count == 0:
            # print("this is the first piont in prior, setting initial energy---------")
            # print("parent initial energy:", self.parent_initial_energy)
            self.prior_initial_energy = V
            # print("prior initial energy:", self.prior_initial_energy)

            V = self.parent_initial_energy
            # print("prior initial energy after subtracting stuff", V)

        else:
            # print("this is NOT the first piont in prior,")
            # print("prior potential", V)
            V -= self.prior_initial_energy
            V += self.parent_initial_energy
            # print("prior potential after subtracting initial energy", V)
        gradV = -self.atoms.get_forces().reshape(-1)
        self.count +=1
        return np.append(np.array(V).reshape(-1), gradV)
