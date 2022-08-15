"""
This file is for setting up and running the tests for this package.
"""

import unittest
from decision_tree_pkg.tests import test_decision_tree

# Intializing the test suite
loader = unittest.TestLoader()
suite = unittest.TestSuite()

# Adding the decision tree tests to the test suite
suite.addTest(loader.loadTestsFromModule(test_decision_tree))

# Creating and initalizing a test runner
runner = unittest.TextTestRunner(verbosity=3)

# Now, we'll run our tests
result = runner.run(suite)
