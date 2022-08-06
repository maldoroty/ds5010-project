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

    def leaf(self):
        '''check to see if a leaf node was reached'''
        if self.label != None:
            return True

class DecisionTree:

    def __init__(self, min_split = 2, max_depth = None, total_features = None):
        '''initiliazing variables that control the decision tree builder'''
        self.root = None
        self.min_split = min_split
        max_depth = max_depth
    
    def start_tree(self, data, col_labels):
        '''starting the tree at the root'''
        self.total_features = data.shape[1]
        self.root = self.build_tree(data, col_labels)
    
    def build_tree(self, data, col, depth):
        '''building the tree'''
        samples, features = np.shape(data)
        diff_labels = np.df['labels'].nunique()
        # stopping condition
        if diff_labels <= 1 or depth >= self.max_depth:
            return Node(label = self.count_leaf(data))
        # need to add traversal of each column element to find best information gain to build tree recurisvely
        best_split = self.best_split(data, col)
        # need to split the data here
        # return decision node from dictionary best_split to fill out arguments
        return Node()
    
    def best_split(data, col):
        pass

    def count_leaf(self, col_labels):
        return max(list(col_labels), key = col_labels.count)