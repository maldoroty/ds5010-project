"""
Unit testing file for decision_tree
"""

import unittest
from decision_tree import *

class TestNode(unittest.TestCase):
    
    def test_node(self):
        x = Node(feature = None, threshold = None, left = None, right = None, label = None)
        self.assertEqual(None, x.feature)
        self.assertEqual(None, x.threshold)
        self.assertEqual(None, x.left)
        self.assertEqual(None, x.right)
        self.assertEqual(None, x.label)

        y = Node(feature = "Color", threshold = "Red")
        self.assertEqual("Color", y.feature)
        self.assertEqual("Red", y.threshold)

        z = Node(label = "Grape")
        self.assertEqual("Grape", z.label)
    
    def test_equal(self):
        x = Node(feature = None, threshold = None, left = None, right = None, label = None)
        y = Node(feature = "Color", threshold = "Red")
        z = Node(label = "Grape")
        self.assertEqual(x, Node(feature = None, threshold = None, label = None))
        self.assertEqual(y, Node(feature = "Color", threshold = "Red", label = None))
        self.assertNotEqual(z, Node(label = "Apple"))

class TestDecisionTree(unittest.TestCase):

    def test_build_tree(self):
        col_names = ["color", "diameter", "labels"]
        df = pd.read_csv("fruits.csv",skiprows=1, header=None, names=col_names)
        labels = df.labels
        df = df.drop(columns = "labels")
        is_numerical = pd.Series([False, True], col_names[:2])
        x = DecisionTree().build_tree(df, labels, is_numerical)
        self.assertEqual(x, Node(feature = "color", threshold = "Red", label = None))
        self.assertEqual(x.left, Node(label = "Grape"))
        self.assertEqual(x.right, Node(label = "Apple"))

        col_names = []
        is_numerical = pd.Series([], col_names)
        y = DecisionTree().build_tree([], labels, is_numerical)
        self.assertEqual(y, None)

class TestDecisionTreeFunctions(unittest.TestCase):

    def test_best_split_df(dataframe: pd.DataFrame, labels, is_numerical: pd.Series):

if __name__ == "__main__":
    unittest.main(verbosity=3)