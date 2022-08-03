'''
DS5010 Decision Tree Project
'''

import pandas as pd

class Node:
    
    def __init__(self, data, left = None, right = None):
        self.data = data
        self.left = left
        self.right = right
    
    def get_data(self):
        return self.data

    def set_data(self, data):
        self.data = data