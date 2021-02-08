# tests/light_test_suite.py
import unittest

# import test modules
from al_mlp.tests.oal_CuNP_case import oal_CuNP

# import and set executor client
from dask.distributed import Client, LocalCluster
from al_mlp.ensemble_calc import EnsembleCalc

cluster = LocalCluster(n_workers=4, processes=True, threads_per_worker=1)
client = Client(cluster)
EnsembleCalc.set_executor(client)

# initialize the test suite
loader = unittest.TestLoader()
suite = unittest.TestSuite()

# add tests to the test suite
suite.addTests(loader.loadTestsFromModule(oal_CuNP))

# Deprecated below, call using pytest instead
# initialize a runner, pass it your suite and run it
# runner = unittest.TextTestRunner(verbosity=3)
# if __name__ == "__main__":
#     result = runner.run(suite)
