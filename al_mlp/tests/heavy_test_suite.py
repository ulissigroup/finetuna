# tests/heavy_test_suite.py
import unittest

# import test modules
from al_mlp.tests.oal_CuNP_case import oal_CuNP

# from al_mlp.tests.oal_PtNP_case import oal_PtNP


# initialize the test suite
loader = unittest.TestLoader()
suite = unittest.TestSuite()

# add tests to the test suite
suite.addTests(loader.loadTestsFromTestCase(oal_CuNP))
# suite.addTests(loader.loadTestsFromTestCase(oal_PtNP))
# add more tests here

# Deprecated below, call using pytest instead
# initialize a runner, pass it your suite and run it
# runner = unittest.TextTestRunner(verbosity=3)
# if __name__ == "__main__":
#     result = runner.run(suite)
