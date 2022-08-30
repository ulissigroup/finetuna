# tests/suites/light_suite.py
import unittest

# import test modules
from finetuna.tests.cases.online_ft_uncertainty_CuNP_test import (
    online_ft_uncertainty_CuNP,
)
from finetuna.tests.cases.online_ft_gemnet_dT_CuNP_test import online_ft_gemnet_dT_CuNP
from finetuna.tests.cases.online_ft_gemnet_oc_CuNP_test import online_ft_gemnet_oc_CuNP

# initialize the test suite
loader = unittest.TestLoader()
suite = unittest.TestSuite()

# add tests to the test suite
suite.addTests(loader.loadTestsFromModule(online_ft_uncertainty_CuNP))
suite.addTests(loader.loadTestsFromModule(online_ft_gemnet_dT_CuNP))
suite.addTests(loader.loadTestsFromModule(online_ft_gemnet_oc_CuNP))
