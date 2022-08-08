'''
DS5010 Decision Tree Project
'''
import pandas as pd
import numpy as np
from collections import Counter

class Node:
    
    def __init__(self, feature = None, threshold = None, left = None, right = None, label = None):
        '''initializing all the variables needed for a decision node and leaf node'''
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label

    def leaf(self):
        '''check to see if a leaf node was reached'''
        if self.label != None:
            return True

class DecisionTree:

    def __init__(self, min_split = 2, max_depth = 10, total_features = None):
        '''initiliazing variables that control the decision tree builder'''
        self.root = None
        self.min_split = min_split
        self.max_depth = max_depth
    
    def start_tree(self, data, col_labels):
        '''starting the tree at the root'''
        self.root = self.build_tree(data, col_labels)
    
    def build_tree(self, data, col_labels, depth = 0):
        '''building the tree'''
        samples, features = data.shape()
        diff_labels = df['labels'].nunique()
        # stopping condition
        if diff_labels <= 1 or depth >= self.max_depth:
            return Node(label = self.count_leaf(data))
        # need to add traversal of each column element to find best information gain to build tree recurisvely
        best_split = self.best_split(data, col_labels)
        # need to split the data here
        # return decision node from best_split to fill out arguments
        return Node()

    def count_leaf(self, col_labels):
        count = Counter(col_labels)
        return count.most_common(1)[0][0]

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
        left_impurity = gini_impurity_pd(left_labels)
        right_impurity = gini_impurity_pd(right_labels)
        # calculate weighted impurity
        weighted_impurity = count * left_impurity + (1 - count) * right_impurity
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

def split(dataframe: pd.DataFrame, col_name, threshold):
    '''split the dataframe based on the threshold in a column'''
    return dataframe[dataframe[col_name] == threshold], dataframe[dataframe[col_name] != threshold]

def get_counts():
    """
        getting the counts of the labels in the dataset, then creating a dict
        with the label with it's count
    """
    # create a list of labels without duplicates
    unique_rows = list(set(self.df.iloc[:,0]))

    # rename dicts
    dicts = {}
    
    for items in unique_rows:
        # making a list of the counts of the labels, will be the values
        counts = self.df[items].value_counts()
        # make dictionary with labels (keys) and their counts (values)
        dicts[i] = counts

    return dicts

def gini_imp(classes):
    """
        performing gini formula on data set, returning the gini impurity
    """
    # find total number of rows in dataset
    total_outcomes = len(classes)
    dicts = get_counts()
    gini_imp = 1

    #perform gini calculation
    for i in dicts:
        values = dicts[i]
        p = values / total_outcomes
        gini = p^2
        gini_imp -= gini
        
    return gini_imp

def weighted_gini_impurity(left, right):
    """
        finding weighted gini value, returning the weighted gini impurity
    """
    #gini from left node
    left_gini = gini_imp(left[0])

    #gini from right node
    right_gini = gini_imp(right[0])

    # weighted gini impurity
    weighted_gini = (left_gini * (len(left_gini) / (len(left_gini) + (len(right_gini))))) + \
                    (right_gini * (len(right_gini) / (len(left_gini) + (len(right_gini)))))

    return weighted_gini

if __name__ == "__main__":
    # example of how this needs to be set up
    col_names = ["color", "diameter", "labels"]
    df = pd.read_csv("fruits.csv", skiprows=1, header=None, names=col_names)
    labels = df.labels
    df = df.drop(columns = "labels")
    # user must specifiy the pd.Series for is_numerical and here the slice eliminates the label column of fruits.csv
    is_numerical = pd.Series([False, True], col_names[:2])
    print(best_split_df(df, labels, is_numerical))
