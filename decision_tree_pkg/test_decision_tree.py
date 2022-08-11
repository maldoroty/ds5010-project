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

    def test_best_split_df(self):
        col_names = ["color", "diameter", "labels"]
        df = pd.read_csv("fruits.csv",skiprows=1, header=None, names=col_names)
        labels = df.labels
        df = df.drop(columns = "labels")
        is_numerical = pd.Series([False, True], col_names[:2])
        x = best_split_df(df, labels, is_numerical)
        self.assertEqual(x, ("color", "Red"))

        col_names2 = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
        df2 = pd.read_csv("iris.csv", skiprows=1, header=None, names=col_names2)
        species = df2.species
        df2 = df2.drop(columns = "species")
        is_numerical2 = pd.Series([True, True, True, True], col_names2[:4])
        y = best_split_df(df2, species, is_numerical2)
        self.assertEqual(y, ("petal_length", 1.9))

        z = best_split_df([], [], [])
        self.assertEqual(z, (None, None))


if __name__ == "__main__":
    unittest.main(verbosity=3)