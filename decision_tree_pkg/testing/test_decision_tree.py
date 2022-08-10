"""
Unit testing file for decision_tree
"""

import unittest
from ..decision_tree import *


class TestDecisionTree(unittest.TestCase):

    def test_build_tree(self):
        df = pd.read_csv("fruits.csv",skiprows=1, header=None, names=col_names)
        col_names = ["color", "diameter", "labels"]
        labels = df.labels
        df = df.drop(columns = "labels")
        is_numerical = pd.Series([False, True], col_names[:2])
        x = DecisionTree().build_tree(df, labels, is_numerical)
        self.assertEqual(x == Node(feature = "color", threshold = "Red", label = None))
        self.assertEqual(x.left == Node("Grape"))
        self.assertEqual(x.right == Node("Apple"))

if __name__ == "__main__":
    unittest.main(verbosity=3)