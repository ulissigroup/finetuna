# tests/heavy_test_suite.py
import unittest

# import test modules
from al_mlp.tests.oal_CuNP_case import oal_CuNP

# from al_mlp.tests.oal_PtNP_case import oal_PtNP


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

# add methods to test case classes
# oal_CuNP.addClassCleanup(plot_forces_hist)
# oal_PtNP.addClassCleanup(plot_forces_hist)

# load test case classes
CuNP_suite = loader.loadTestsFromTestCase(oal_CuNP)
# PtNP_suite = loader.loadTestsFromTestCase(oal_PtNP)

# add cleanup methods to test cases
CuNP_suite._tests[0].addCleanup(plot_forces_hist, CuNP_suite._tests[0])
# PtNP_suite._tests[0]._tests[0].addCleanup(plot_forces_hist)

# add tests cases to the test suite
suite.addTests(CuNP_suite)
# suite.addTests(PtNP_suite)

# initialize a runner, pass it your suite and run it
runner = unittest.TextTestRunner(verbosity=3)
if __name__ == "__main__":
    result = runner.run(suite)
