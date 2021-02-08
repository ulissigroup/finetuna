# tests/heavy_test_suite.py
import unittest

# import test modules
# from al_mlp.tests.oal_PtNP_case import oal_PtNP
from al_mlp.tests.case_oal_CuNP import oal_CuNP


# import make_ensemble and dask for setting parallelization
from al_mlp.ensemble_calc import EnsembleCalc
from dask.distributed import Client, LocalCluster

# Set dask client in ensemble calc
if __name__ == "__main__":
    cluster = LocalCluster(processes=True, threads_per_worker=1)
    client = Client(cluster)
    EnsembleCalc.set_executor(client)

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
