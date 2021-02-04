# tests/light_test_suite.py
import unittest

# import test modules
from al_mlp.tests.oal_CuNP_case import oal_CuNP


# initialize the test suite
loader = unittest.TestLoader()
suite = unittest.TestSuite()

# add tests to the test suite
suite.addTests(loader.loadTestsFromModule(oal_CuNP))

# initialize a runner, pass it your suite and run it
runner = unittest.TextTestRunner(verbosity=3)
if __name__ == "__main__":
    result = runner.run(suite)
