'''
DS5010 Decision Tree Project
'''
import pandas as pd
import numpy as np

class Node:
    
    def __init__(self, feature = None, threshold = None, left = None, right = None, label = None):
        '''initializing all the variables needed for a decision node and leaf node'''
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label

class DecisionTree:

    def __init__(self, max_depth = 10):
        '''initiliazing variables that control the decision tree builder'''
        self.max_depth = max_depth
    
    def build_tree(self, dataframe, labels, is_numerical, depth = 0):
        """building the tree recursively"""
        if len(dataframe) == 0:
            return None
        if labels.nunique() == 1 or depth >= self.max_depth:
            return Node(label = labels.mode()[0])
        feature, threshold = best_split_df(dataframe, labels, is_numerical)
        left, right, left_labels, right_labels = split(dataframe, feature, threshold, labels, is_numerical)
        left_node = self.build_tree(left, left_labels, is_numerical, depth + 1)
        right_node = self.build_tree(right, right_labels, is_numerical, depth + 1)
        return Node(feature, threshold, left_node, right_node)

def predict(node, input_data):
    """ prediction method"""

    if node.label is not None:
        return node.label
    else:
        if input_data.loc[node.feature] < node.threshold:
            return predict(node.left, input_data)
        else:
            return predict(node.right, input_data)
    

def best_split_df(dataframe: pd.DataFrame, labels, is_numerical: pd.Series):
    """looping through each column of the dataframe"""
    best_threshold = None
    best_impurity = np.inf
    for col_name in dataframe:
        threshold, impurity = best_split_col(dataframe[col_name], labels, is_numerical[col_name])
        # saving lowest impurity
        if impurity < best_impurity:
            best_impurity, best_threshold, col = impurity, threshold, col_name

    return col, best_threshold

def best_split_col(data: pd.Series, labels: pd.Series, is_numerical: bool):
    """to find the best split of the column based on the gini_impurity"""
    # getting relative frequences of each unique values
    unique_features = data.value_counts(normalize = True)
    impurity = np.inf
    # loop through the column getting the value and the count
    for value, count in unique_features.iteritems():
        # if the column is categorical split each row, left is same, right is different
        if not is_numerical:
            left_labels = labels[data == value]
            right_labels = labels[data != value]
        # if the column is numerical, left is less than or equal to, right is greater
        else:
            left_labels = labels[data <= value]
            right_labels = labels[data > value]
        # calculate left and right impurity
        
        #left_impurity = gini_impurity_pd(left_labels)
        
        #right_impurity = gini_impurity_pd(right_labels)
        
        # calculate weighted impurity
        #weighted_impurity = count * left_impurity + (1 - count) * right_impurity

        weighted_impurity = weighted_gini_impurity(left_labels, right_labels)
        
        # save the best weighted impurity and threshold
        if weighted_impurity < impurity:
            impurity = weighted_impurity
            threshold = value
    return threshold, impurity


def gini_impurity_pd(label_subsets: pd.Series) -> float:
    '''fine gini impurity'''
    # get the relative frequencies of the label_subsets
    label_subsets = label_subsets.value_counts(normalize = True)
    # calculate gini impurity
    squared_freq = label_subsets ** 2
    sum_freq = sum(squared_freq)
    return 1 - sum_freq

def split(dataframe: pd.DataFrame, col_name, threshold, labels, is_numerical):
    '''split the dataframe based on the threshold in a column'''
    col_names = list(dataframe.columns)
    idx = col_names.index(col_name)
    if is_numerical[idx]:
        return dataframe[dataframe[col_name] <= threshold], dataframe[dataframe[col_name] > threshold], labels[dataframe[col_name] <= threshold], labels[dataframe[col_name] > threshold]
    return dataframe[dataframe[col_name] == threshold], dataframe[dataframe[col_name] != threshold], labels[dataframe[col_name] == threshold], labels[dataframe[col_name] != threshold]


def build_dt(dataframe: pd.DataFrame, node, labels, is_numerical, depth, col_name, threshold):
    """ building tree"""
    parent = split(dataframe, col_name, threshold, labels, is_numerical)
    built_tree = DecisionTree(max_depth)
    built_tree.build_tree(dataframe, labels, is_numerical, 1)

def gini_imp(classes):
    """
        performing gini formula on data set, returning the gini impurity
    """
    # find total number of rows in dataset
    total_outcomes = len(classes)
    # Maybe should set normalize as True
    counts = dict(classes.value_counts())
    gini_imp = 1

    #perform gini calculation
    for i in counts:
        values = counts[i]
        p = values / total_outcomes
        gini = p ** 2
        gini_imp -= gini
        
    return gini_imp

def weighted_gini_impurity(left, right):
    """
        finding weighted gini value, returning the weighted gini impurity
    """
    #gini from left node
    left_gini = gini_imp(left)

    #gini from right node
    right_gini = gini_imp(right)

    # weighted gini impurity
    weighted_gini = (left_gini * (len(left) / (len(left) + (len(right))))) + \
                    (right_gini * (len(right) / (len(left) + (len(right)))))

    return weighted_gini

TREE_SPACE = "   "

def print_tree_helper(node):
    if node.label is not None:
        tree = "<Leaf, Label = " + node.label + ">"
    else:
        tree = "<Node, Threshold = " + node.threshold
        tree += ", Features = " + node.feature + ">\n"

        tree += TREE_SPACE + TREE_SPACE + print_tree(node.left) + "\n"
        tree += TREE_SPACE + TREE_SPACE + print_tree(node.right)

    return tree

def print_tree(node):
    return print_tree_helper(node)
    

if __name__ == "__main__":
    # example of how this needs to be set up
    col_names = ["color", "diameter", "labels"]
    df = pd.read_csv("fruits.csv", skiprows=1, header=None, names=col_names)
    labels = df.labels
    print(df)
    df = df.drop(columns = "labels")
    # user must specifiy the pd.Series for is_numerical and here the slice eliminates the label column of fruits.csv
    is_numerical = pd.Series([False, True], col_names[:2])
    
    print(best_split_df(df, labels, is_numerical))

    # Original split = ('color', 'Red')

    

    tree = DecisionTree()
    node = tree.build_tree(df, labels, is_numerical)
    print(print_tree(node))
    # print(predict(node, df))

    
 
