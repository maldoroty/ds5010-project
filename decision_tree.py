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
        diff_labels = len(np.df['labels'].nunique())
        # stopping condition
        if diff_labels <= 1 or depth >= self.max_depth:
            return Node(label = self.count_leaf(data))
        # need to add traversal of each column element to find best information gain to build tree recurisvely
        best_split = self.best_split(data, col_labels)
        # need to split the data here
        # return decision node from best_split to fill out arguments
        return Node()
    
    def best_split(self, data, col):
        pass
    
    def split(self, data, index, threshold):
        pass

    def count_leaf(self, col_labels):
        count = Counter(col_labels)
        return count.most_common(1)[0][0]