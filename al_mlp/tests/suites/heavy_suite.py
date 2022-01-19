# tests/suites/heavy_suite.py
import unittest

# import test modules
# from al_mlp.tests.cases.online_PtNP_test import online_PtNP
from al_mlp.tests.cases.offline_neb_CuC_test import offline_NEB
from al_mlp.tests.cases.online_CuNP_test import online_CuNP
from al_mlp.tests.cases.offline_CuNP_test import offline_CuNP
from al_mlp.tests.cases.online_flare_CuNP_test import online_flare_CuNP
from al_mlp.tests.cases.delta_CuNP_test import delta_CuNP
from al_mlp.tests.cases.offline_uncertainty_CuNP_test import offline_uncertainty_CuNP
from al_mlp.tests.cases.online_ft_CuNP_test import online_ft_CuNP

# import make_ensemble and dask for setting parallelization
from al_mlp.ml_potentials.amptorch_ensemble_calc import AmptorchEnsembleCalc
from dask.distributed import Client, LocalCluster

# Set dask client in ensemble calc
if __name__ == "__main__":
    cluster = LocalCluster(processes=True, threads_per_worker=1)
    client = Client(cluster)
    AmptorchEnsembleCalc.set_executor(client)

# initialize the test suite
loader = unittest.TestLoader()
suite = unittest.TestSuite()

# add tests to the test suite
suite.addTests(loader.loadTestsFromTestCase(offline_CuNP))
suite.addTests(loader.loadTestsFromTestCase(online_CuNP))
suite.addTests(loader.loadTestsFromTestCase(offline_NEB))
suite.addTests(loader.loadTestsFromModule(online_flare_CuNP))
suite.addTests(loader.loadTestsFromTestCase(delta_CuNP))
suite.addTests(loader.loadTestsFromTestCase(offline_uncertainty_CuNP))
suite.addTests(loader.loadTestsFromTestCase(online_ft_CuNP))
# suite.addTests(loader.loadTestsFromTestCase(online_PtNP))
# add more tests here

# Deprecated below, call using pytest instead
# initialize a runner, pass it your suite and run it
# runner = unittest.TextTestRunner(verbosity=3)
# if __name__ == "__main__":
#     result = runner.run(suite)
