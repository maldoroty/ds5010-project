'''
DS5010 Decision Tree Project
'''

import pandas as pd
import numpy as np

class Node:
    
    def __init__(self, category = None, split_value = None, left = None, right = None, gini_index = None, label = None):
        self.category = category
        self.split_value = split_value
        self.left = left
        self.right = right
        self.gini_index = gini_index
        self.label = label

class DecisionTree:

    def __init__(self):
        pass

    def build_tree(self, data):
        
        diff_labels = np.df['labels'].nunique()
        if diff_labels > 1:
            best_split = self.best_split(data)
            left, right = self.build_tree.split(data)
            return Node(best_split["category"], best_split["split_value"], left, right, best_split["gini_index"])
        
        return Node(label = self.count_leaf(data))

    def best_split(data):
        pass

    def split(data):
        pass
    
    def count_leaf(data):
        pass