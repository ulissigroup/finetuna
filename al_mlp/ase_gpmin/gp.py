import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve_triangular, cho_factor, cho_solve
from ase.optimize.gpmin.kernel import SquaredExponential
from ase.optimize.gpmin.prior import ZeroPrior


class GaussianProcess():
    """Gaussian Process Regression
    It is recommended to be used with other Priors and Kernels from
    ase.optimize.gpmin

    Parameters:

    prior: Prior class, as in ase.optimize.gpmin.prior
           Defaults to ZeroPrior

    kernel: Kernel function for the regression, as in
            ase.optimize.gpmin.kernel
            Defaults to the Squared Exponential kernel with derivatives
    """
    def __init__(self, prior=None, kernel=None):
        if kernel is None:
            self.kernel = SquaredExponential()
        else:
            self.kernel = kernel

        if prior is None:
            self.prior = ZeroPrior()
        else:
            self.prior = prior

    def set_hyperparams(self, params):
        """Set hyperparameters of the regression.
        This is a list containing the parameters of the
        kernel and the regularization (noise)
        of the method as the last entry.
        """
        self.hyperparams = params
        self.kernel.set_params(params[:-1])
        self.noise = params[-1]

    def train(self, X, Y, noise=None):
        """Produces a PES model from data.

        Given a set of observations, X, Y, compute the K matrix
        of the Kernel given the data (and its cholesky factorization)
        This method should be executed whenever more data is added.

        Parameters:

        X: observations (i.e. positions). numpy array with shape: nsamples x D
        Y: targets (i.e. energy and forces). numpy array with
            shape (nsamples, D+1)
        noise: Noise parameter in the case it needs to be restated.
        """
        if noise is not None:
            self.noise = noise  # Set noise attribute to a different value

        self.X = X.copy()  # Store the data in an attribute
        n = self.X.shape[0]
        D = self.X.shape[1]
        regularization = np.array(n * ([self.noise * self.kernel.l] +
                                  D * [self.noise]))

        K = self.kernel.kernel_matrix(X)  # Compute the kernel matrix
        K[range(K.shape[0]), range(K.shape[0])] += regularization**2

        self.m = self.prior.prior(X)
        self.a = Y.flatten() - self.m
        self.L, self.lower = cho_factor(K, lower=True, check_finite=True)
        cho_solve((self.L, self.lower), self.a, overwrite_b=True,
                  check_finite=True)

    def predict(self, x, get_variance=False):
        """Given a trained Gaussian Process, it predicts the value and the
        uncertainty at point x. It returns f and V:
        f : prediction: [y, grady]
        V : Covariance matrix. Its diagonal is the variance of each component
            of f.

        Parameters:

        x (1D np.array):      The position at which the prediction is computed
        get_variance (bool):  if False, only the prediction f is returned
                              if True, the prediction f and the variance V are
                              returned: Note V is O(D*nsample2)
        """
        n = self.X.shape[0]
        k = self.kernel.kernel_vector(x, self.X, n)
        f = self.prior.prior(x) + np.dot(k, self.a)
        if get_variance:
            v = solve_triangular(self.L, k.T.copy(), lower=True,
                                 check_finite=False)
            variance = self.kernel.kernel(x, x)
            # covariance = np.matmul(v.T, v)
            covariance = np.tensordot(v, v, axes=(0, 0))
            V = variance - covariance
            return f, V
        return f

    def neg_log_likelihood(self, params, *args):
        """Negative logarithm of the marginal likelihood and its derivative.
        It has been built in the form that suits the best its optimization,
        with the scipy minimize module, to find the optimal hyperparameters.

        Parameters:

        l: The scale for which we compute the marginal likelihood
        *args: Should be a tuple containing the inputs and targets
               in the training set-
        """
        X, Y = args
        # Come back to this
        self.kernel.set_params(np.array([params[0], params[1], self.noise]))
        self.train(X, Y)
        y = Y.flatten()

        # Compute log likelihood
        logP = (-0.5 * np.dot(y - self.m, self.a) -
                np.sum(np.log(np.diag(self.L))) -
                X.shape[0] * 0.5 * np.log(2 * np.pi))

        # Gradient of the loglikelihood
        grad = self.kernel.gradient(X)

        # vectorizing the derivative of the log likelihood
        D_P_input = np.array([np.dot(np.outer(self.a, self.a), g)
                              for g in grad])
        D_complexity = np.array([cho_solve((self.L, self.lower), g)
                                 for g in grad])

        DlogP = 0.5 * np.trace(D_P_input - D_complexity, axis1=1, axis2=2)
        return -logP, -DlogP

    def fit_hyperparameters(self, X, Y, tol=1e-2, eps=None):
        """Given a set of observations, X, Y; optimize the scale
        of the Gaussian Process maximizing the marginal log-likelihood.
        This method calls TRAIN there is no need to call the TRAIN method
        again. The method also sets the parameters of the Kernel to their
        optimal value at the end of execution

        Parameters:

        X:   observations(i.e. positions). numpy array with shape: nsamples x D
        Y:   targets (i.e. energy and forces).
             numpy array with shape (nsamples, D+1)
        tol: tolerance on the maximum component of the gradient of the
             log-likelihood.
             (See scipy's L-BFGS-B documentation:
             https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
        eps: include bounds to the hyperparameters as a +- a percentage
             if eps is None there are no bounds in the optimization

        Returns:

        result (dict) :
              result = {'hyperparameters': (numpy.array) New hyperparameters,
                        'converged': (bool) True if it converged,
                                            False otherwise
                       }
        """
        params = np.copy(self.hyperparams)[:2]
        arguments = (X, Y)
        if eps is not None:
            bounds = [((1 - eps) * p, (1 + eps) * p) for p in params]
        else:
            bounds = None

        result = minimize(self.neg_log_likelihood, params, args=arguments,
                          method='L-BFGS-B', jac=True, bounds=bounds,
                          options={'gtol': tol, 'ftol': 0.01 * tol})

        if not result.success:
            converged = False
        else:
            converged = True
            self.hyperparams = np.array([result.x.copy()[0],
                                         result.x.copy()[1], self.noise])
        self.set_hyperparams(self.hyperparams)
        return {'hyperparameters': self.hyperparams, 'converged': converged}
