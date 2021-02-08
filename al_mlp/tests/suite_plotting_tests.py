# tests/heavy_test_suite.py
import unittest

# import test modules
from al_mlp.tests.case_oal_PtNP import oal_PtNP
from al_mlp.tests.case_oal_CuNP import oal_CuNP

# import make_ensemble and dask for setting parallelization
from al_mlp.ensemble_calc import EnsembleCalc
from dask.distributed import Client, LocalCluster


# define extra plotting test code
def plot_forces_hist(self) -> None:
    import matplotlib.pyplot as plt
    import os

    if not os.path.exists("test_figs"):
        os.makedirs("test_figs")

    plt.figure(1)
    plt.hist(self.EMT_image.get_forces().flatten(), bins=100)
    plt.xlim([-0.08, 0.08])
    plt.ylim([0, 4])
    plt.title(self.description + " EMT forces distribution")
    plt.savefig("test_figs/" + self.description + "_EMT_hist")

    plt.figure(2)
    plt.hist(self.OAL_image.get_forces().flatten(), bins=100)
    plt.xlim([-0.08, 0.08])
    plt.ylim([0, 4])
    plt.title(self.description + " OAL forces distribution")
    plt.savefig("test_figs/" + self.description + "_OAL_hist")


# initialize the test suite
loader = unittest.TestLoader()
suite = unittest.TestSuite()

# add methods to test case classes (Only works this way in Python 3.8+)
# oal_PtNP.addClassCleanup(plot_forces_hist)
# oal_CuNP.addClassCleanup(plot_forces_hist)

# load test case classes
PtNP_suite = loader.loadTestsFromTestCase(oal_PtNP)
CuNP_suite = loader.loadTestsFromTestCase(oal_CuNP)

# add cleanup methods to test cases
PtNP_suite._tests[0].addCleanup(plot_forces_hist, PtNP_suite._tests[0])
CuNP_suite._tests[0].addCleanup(plot_forces_hist, CuNP_suite._tests[0])

# add tests cases to the test suite
suite.addTests(PtNP_suite)
suite.addTests(CuNP_suite)

# initialize a runner, pass it your suite and run it
runner = unittest.TextTestRunner(verbosity=3)
if __name__ == "__main__":
    # Set dask client in ensemble calc
    cluster = LocalCluster(processes=True, threads_per_worker=1)
    client = Client(cluster)
    EnsembleCalc.set_executor(client)

    # run
    result = runner.run(suite)
