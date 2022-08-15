"""
Unit testing file for decision_tree
"""

import unittest
from ..decision_tree import *

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
        x = DecisionTree(df).build_tree(df, labels, is_numerical)
        self.assertEqual(x, Node(feature = "color", threshold = "Red", label = None))
        self.assertEqual(x.left, Node(label = "Grape"))
        self.assertEqual(x.right, Node(label = "Apple"))

        col_names = []
        is_numerical = pd.Series([], col_names)
        y = DecisionTree(pd.DataFrame([])).build_tree([], labels, is_numerical)
        self.assertEqual(y, None)

    def test_train(self):
        """
        Testing the train function
        """
        df = pd.read_csv("fruits.csv")
        tree_with_nodes = DecisionTree(df).train()
        self.assertEqual(tree_with_nodes, Node(feature = "color", threshold = "Red", label = None))
        self.assertEqual(tree_with_nodes.left, Node(label = "Grape"))
        self.assertEqual(tree_with_nodes.right, Node(label = "Apple"))

        tree_empty = DecisionTree(pd.DataFrame([])).train()
        self.assertEqual(tree_empty, None)

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


    def test_predict(self):
        """
        Testing prediction function leaf, node with numeric data, and node with string data
        """
        # Predict with leaf
        leaf = Node()
        leaf.label = "example"
        pred = predict(leaf, pd.Series([1, 2, "large"]))
        self.assertEqual(pred, "example")

        # Predict with node with 2 children
        left_leaf = Node()
        left_leaf.label = "left"
        right_leaf = Node()
        right_leaf.label = "right"

        # Numeric data
        numeric_node = Node(feature=0, threshold=2.3, left=left_leaf, right=right_leaf)
        left_pred_numeric = predict(numeric_node, pd.Series([1.2]))
        right_pred_numeric = predict(numeric_node, pd.Series([2.4]))
        self.assertEqual(left_pred_numeric, "left")
        self.assertEqual(right_pred_numeric, "right")

        # String data
        str_node = Node(feature=0, threshold="blue", left=left_leaf, right=right_leaf)
        left_pred_str = predict(str_node, pd.Series(["blue"]))
        right_pred_str = predict(str_node, pd.Series(["red"]))
        self.assertEqual(left_pred_str, "left")
        self.assertEqual(right_pred_str, "right")



    def test_best_split_col(self):
        """
        Testing the function that determines the best split in a given column
        """
        col_names = ["color", "diameter", "labels"]
        df = pd.read_csv("fruits.csv",skiprows=1, header=None, names=col_names)
        labels = df.labels
        df = df
        is_numerical = gen_is_numerical(df)

        # With numeric column
        self.assertEqual(best_split_col(df["diameter"], labels, is_numerical["diameter"]), (1, 0.0))
        
        # With string column
        self.assertEqual(best_split_col(df["color"], labels, is_numerical["color"]), ("Red", 0.0))



    def test_split(self):
        """
        Testing the split function, which splits a given dataframe based on a certain column and threshold
        """
        df = pd.read_csv("fruits.csv")
        labels = df.label
        df = df
        is_numerical = gen_is_numerical(df)


        column = "diameter"
        threshold = 1
        result = split(df.drop(df.columns[-1], axis=1), column, threshold, labels, is_numerical)

        right_df = pd.DataFrame([["Red", 1], ["Red", 1]], columns=df.columns[:len(df.columns) - 1], index=[2,3])
        left_df = pd.DataFrame([["Green", 3], ["Yellow", 3], ["Yellow", 3]], columns=df.columns[:len(df.columns) - 1], index=[0,1,3])
        right_labels = pd.Series(["Grape", "Grape"], index=[2, 3])
        left_labels = pd.Series(["Apple", "Apple", "Apple"], index=[0, 1, 4])
       
        # Have to use sort_index(inplace=True) for the DataFrame comparison to work for some reason
        self.assertEqual(result[0].sort_index(inplace=True), right_df.sort_index(inplace=True))
        self.assertEqual(result[1].sort_index(inplace=True), left_df.sort_index(inplace=True))
        self.assertEqual(result[2].sort_index(inplace=True), right_labels.sort_index(inplace=True))
        self.assertEqual(result[3].sort_index(inplace=True), left_labels.sort_index(inplace=True))



    def test_tree_score(self):
        """
        Testing the tree_score function, which measures the accuracy of a given tree
        """
        # Just a random number to serve as a seed for our RNG. We need so we can have consistent
        # test results when using rand_int().
        RANDOMNESS = 25

        df = pd.read_csv("iris.csv")

        Xy_train, X_test, y_test = train_test_split(df, 0.75, RANDOMNESS)

        tree = DecisionTree(Xy_train).train()

        score = tree_score(tree, X_test, y_test)

        self.assertAlmostEqual(score, 0.89473684)


        


    def test_gen_is_numerical(self):
        """
        Testing the gen_is_numerical function, which generates a series describing whether each column of
        the dataframe is numeric or not.
        """

        df = pd.read_csv("fruits.csv")

        is_numerical_series = gen_is_numerical(df)

        expected_is_numerical = pd.Series([False, True], ["color", "diameter"])

        self.assertEqual(is_numerical_series.sort_index(inplace=True), expected_is_numerical.sort_index(inplace=True))

        


    def test_train_test_split(self):
        """
        Testing the train_test_split, which divides a given dataset into training and testing subsets.
        """
        RANDONMNESS = 25

        df = pd.read_csv("fruits.csv")

        Xy_train, X_test, y_test = train_test_split(df, 0.50, RANDONMNESS)

        expected_Xy_train = pd.DataFrame([["Yellow", 3, "Apple"], ["Red", 1, "Grape"]], columns=["color", "diameter", "label"])
        expected_X_test = pd.DataFrame([["Green", 3], ["Red", 1], ["Yellow", 3]], columns=["color", "diameter"])
        expected_y_test = pd.Series(["Apple", "Grape", "Apple"])

        self.assertEqual(Xy_train.sort_index(inplace=True), expected_Xy_train.sort_index(inplace=True))
        self.assertEqual(X_test.sort_index(inplace=True), expected_X_test.sort_index(inplace=True))
        self.assertEqual(y_test.sort_index(inplace=True), expected_y_test.sort_index(inplace=True))
