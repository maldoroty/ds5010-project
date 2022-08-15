# Simple Decision Tree Classfier using Gini Impurity
The purpose of this package is provide an easy way to use a Decision Tree classifier for people who are new to machine learning and data science. The end goal was to provide a simple and concise implementation of the Decision Tree algorithm that is suitable for anyone new to learning ML. As well, we want the user to leave with a better understanding of how the Decision Tree algorithm works.

This package was created with a focus on accessbility. It should be easy for someone to input their CSV data as seen below to do their classification predictions in whatever setting.
The main objects - the `Node` and the `DecisionTree` classes - are at the top of the `decision_tree.py` file.

As well, there are functions and methods responsible for the necessary dataframe looping, gini impurity calculation, dataframe splitting, training and testing, tree scoring, and more. All of the code was written in an easy-to-understand way.

The decision tree class is the most important, thus it was kept clean and easily readable calling the helper functions that can be seen below. 

## Example Usage
Here's an example of using our package using the famous Iris dataset. The code for the example can also be found in the `example.py` file.
```python
    
    # Training and testing the decision tree using the iris dataset

    # put the iris csv file into a dataframe
    df = pd.read_csv("iris.csv")

    # train_test_split produce Xy_train, X_test, and y_test. Xy_train shows the classes and the data combined into a dataset. The x_test shows the node features while the y_test shows the node labels.
    Xy_train, X_test, y_test = train_test_split(df=df, train_size=0.50)

    tree = DecisionTree(Xy_train)
    node = tree.train()

    # prints out the decision tree to help visualize the tree and leaf nodes
    print(node)

    # this tells the score of how accurate our prediction was
    tree_score(node, X_test, y_test)
```

## Tests
To run our tests, please run the `run_tests.py` file.

## Credits
This package was created by Michael A. and Hajera S. for their final project in the DS5010 class.
