'''
DS5010 Decision Tree Project
'''

#import pandas as pd

class Node:

    def __init__(self, data, prev = None, next = None):
        self.data = data
        self.prev = prev
        self.next = next
    
    def get_data(self):
        return self.data

if __name__ == "__main__":
    x = Node(20)
    print(x.get_data())