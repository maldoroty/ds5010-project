import pandas as pd
from decision_tree_pkg.decision_tree import *

"""
Example usage of the decision class package.
"""

# Read in a CSV as a pandas DataFrame
df = pd.read_csv("iris.csv")

# Create our testing and training datasets
Xy_train, X_test, y_test = train_test_split(df=df, train_size=0.50)

# Printing out the datasets to see how they look like
print("Xy_train = \n", Xy_train)
print("X_test = \n", X_test)
print("y_test = \n", y_test)

# Creating a DecisionTree and training it
tree = DecisionTree(Xy_train)
node = tree.train()
print(node)

# Calculating the tree's score using the testing dataset
score = tree_score(node, X_test, y_test)
print(score)  
    
