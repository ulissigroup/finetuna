"""
Test script to be executed before pushing or submitting a PR to master
repository.
"""

import unittest

from .delta_energy_test import test_delta_sub, test_delta_add

class TestMethods(unittest.TestCase):
    def test_delta_sub(self):
        test_delta_sub()

    def test_delta_add(self):
        test_delta_add()

if __name__ == "__main__":
    unittest.main(warnings="ignore")
