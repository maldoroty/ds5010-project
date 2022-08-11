"""
Unit testing file for decision_tree
"""

import unittest
from decision_tree import *

class TestNode(unittest.TestCase):
    
    def test_node(self):
        """
        Testing node attribute retrieval
        """
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
        """
        Testing node equality
        """
        x = Node(feature = None, threshold = None, left = None, right = None, label = None)
        y = Node(feature = "Color", threshold = "Red")
        z = Node(label = "Grape")
        self.assertEqual(x, Node(feature = None, threshold = None, label = None))
        self.assertEqual(y, Node(feature = "Color", threshold = "Red", label = None))
        self.assertNotEqual(z, Node(label = "Apple"))

class TestDecisionTree(unittest.TestCase):

    def test_build_tree(self):
        """
        testing the build tree function
        """
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
        """
        Test best split
        """
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
    
    def test_gini_imp(self):
        """
        Testing gini_impurity of one series
        """
        series_1 = pd.Series(["Apple", "Apple", "Apple"])
        series_2 = pd.Series(["Apple", "Apple", "Grape"])
        series_3 = pd.Series(["Apple", "Grape"])
        x = gini_imp(series_1)
        y = gini_imp(series_2)
        z = gini_imp(series_3)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 1 - ((2/3) ** 2 + (1/3) ** 2))
        self.assertAlmostEqual(z, 0.5)
    
    def test_weighted_gini_imp(self):
        """
        Testing weighted gini impurity of two different series
        """
        x = series_1 = pd.Series(["Apple", "Apple", "Apple"])
        y = series_2 = pd.Series(["Apple", "Apple", "Grape"])
        z = series_3 = pd.Series(["Apple", "Grape"])
        weight_1 = weighted_gini_impurity(x, y)
        weight_2 = weighted_gini_impurity(x, z)
        weight_3 = weighted_gini_impurity(x, pd.Series(["Grape", "Grape"]))
        self.assertAlmostEqual(weight_1, ((1/2) * 0) + ((1/2) * 0.4444444))
        self.assertAlmostEqual(weight_2, (3/5) * 0 + (2/5) * 0.5)
        self.assertAlmostEqual(weight_3, (3/5) * 0 + (2/5) * 0)


if __name__ == "__main__":
    unittest.main(verbosity=3)