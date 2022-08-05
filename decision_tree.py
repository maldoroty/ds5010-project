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