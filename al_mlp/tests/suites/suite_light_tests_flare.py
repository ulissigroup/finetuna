# tests/suite_light_tests.py
import unittest

# import test modules
from al_mlp.tests.cases.case_online_CuNP_flare import online_CuNP

# from al_mlp.tests.cases.case_offline_CuNP_flare import offline_CuNP

# initialize the test suite
loader = unittest.TestLoader()
suite = unittest.TestSuite()

# add tests to the test suite
suite.addTests(loader.loadTestsFromModule(online_CuNP))
# suite.addTests(loader.loadTestsFromTestCase(offline_CuNP))