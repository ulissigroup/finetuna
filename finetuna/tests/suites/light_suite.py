# tests/suites/light_suite.py
import unittest

# import test modules
from finetuna.tests.cases.delta_CuNP_test import delta_CuNP
from finetuna.tests.cases.online_CuNP_test import online_CuNP
from finetuna.tests.cases.offline_CuNP_test import offline_CuNP
from finetuna.tests.cases.offline_uncertainty_CuNP_test import offline_uncertainty_CuNP

# from finetuna.tests.cases.online_flare_CuNP_test import online_flare_CuNP
from finetuna.tests.cases.online_ft_CuNP_test import online_ft_CuNP

# import and set executor client
from dask.distributed import Client, LocalCluster
from finetuna.ml_potentials.amptorch_ensemble_calc import AmptorchEnsembleCalc

cluster = LocalCluster(n_workers=4, processes=True, threads_per_worker=1)
client = Client(cluster)
AmptorchEnsembleCalc.set_executor(client)

# initialize the test suite
loader = unittest.TestLoader()
suite = unittest.TestSuite()

# add tests to the test suite
suite.addTests(loader.loadTestsFromModule(online_CuNP))
suite.addTests(loader.loadTestsFromTestCase(offline_CuNP))
suite.addTests(loader.loadTestsFromModule(offline_uncertainty_CuNP))
suite.addTests(loader.loadTestsFromModule(delta_CuNP))

# suite.addTests(loader.loadTestsFromModule(online_flare_CuNP))
suite.addTests(loader.loadTestsFromModule(online_ft_CuNP))

# Deprecated below, call using pytest instead
# initialize a runner, pass it your suite and run it
# runner = unittest.TextTestRunner(verbosity=3)
# if __name__ == "__main__":
#     result = runner.run(suite)
