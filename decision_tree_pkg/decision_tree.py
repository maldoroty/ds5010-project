'''
DS5010 Decision Tree Project
'''
import pandas as pd
import numpy as np
import random

TREE_SPACE = "   "

class Node:
    """
    Node used to build a decision tree
    """
    def __init__(self, feature = None, threshold = None, left = None, right = None, label = None):
        """
        Initializing all the features of the node

        ...
        
        Attributes
        ----------
        feature : String
            column name where the threshold is picked for best split
        threshold : Float, Int, or String
            value used for the best split
        left : Node
            left child Node
        right : Node
            right child Node
        label : Float, Int, or String
            leaf node most common occuring classifier
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label
    
    def __str__(self):
        """
        Used to print the decision tree 
        """
        
        return self.print_helper(TREE_SPACE)
        
    
    def print_helper(self, space):
        """
        Helper function to print the decision tree

        ...
        
        Attributes
        ----------
        space : String
            spacing that will help with the layout when printing the tree

        ...
        
        Returns
        -------
        string
            printed tree with levels of node and leaf shown
        """
        
        if self.label is not None:
            tree = "<Leaf, Label = " + str(self.label) + ">"
        else:
            tree = "<Node, Threshold = " + str(self.threshold)
            tree += ", Features = " + str(self.feature) + ">\n"
            tree += space + self.left.print_helper(space + TREE_SPACE) + "\n"
            tree += space + self.right.print_helper(space + TREE_SPACE)

        return tree
    
    def __eq__(self, other):
        """
        Equality for two nodes
        """
        return self.feature == other.feature and self.threshold == other.threshold and self.label == other.label

class DecisionTree:
    """
    Decision tree class that uses the nodes and helper functions to build the tree
    """
    
    def __init__(self, df, max_depth = 10):
        """
        Initializing the Tree

        ...
        
        Attributes
        ----------
        max_depth : Int
            The max depth allowed for a tree to be built.
        df : Pandas dataframe
            Dataframe that will be used to train the tree.
        """
        self.max_depth = max_depth
        self.df = df

        
    def train(self):
        """
        Method to prepare the data and start the process to give to the decision tree

        ...
        
        Attributes
        ----------
        None
        
        ...

        Returns
        -------
        Produces a fully built decision tree
        """
        if len(self.df) == 0:
            return None
        
        labels = self.df[self.df.columns[-1]]
        df_no_labels = self.df.drop(self.df.columns[-1], axis=1)
        is_numerical = gen_is_numerical(df_no_labels)

        return self.build_tree(df_no_labels, labels, is_numerical)
    
    def build_tree(self, dataframe, labels, is_numerical, depth = 0):
        """
        Method to build the decision tree

        ...
        
        Attributes
        ----------
        dataframe : pd.Dataframe
            Dataframe used for training the tree
        labels : pd.Series
            Series of labels for training data that gives each row a classification
        is_numerical : pd.Series(list, list)
            Series of a tuple of two lists. The first list is a list of booleans telling if the column is numerical or categorical.
            The second list in the tuple are the column names except for the last (the labels column)
        
        ...

        Returns
        -------
        Entirely built decision tree object
        """
        # base cases to stop building the tree return leaf nodes or None
        if len(dataframe) == 0:
            return None
        if labels.nunique() == 1 or depth >= self.max_depth:
            return Node(label = labels.mode()[0])
        # finding the best split
        feature, threshold = best_split_df(dataframe, labels, is_numerical)
        # splitting the dataframe
        left, right, left_labels, right_labels = split(dataframe, feature, threshold, labels, is_numerical)
        # recursively build the tree adding 1 to the depth
        left_node = self.build_tree(left, left_labels, is_numerical, depth + 1)
        right_node = self.build_tree(right, right_labels, is_numerical, depth + 1)
        return Node(feature, threshold, left_node, right_node)
    
def predict(node, input_data):
    """
    Predict the label of a given data set

    ...

    Paramters
    ---------
    Node : Node
        The tree root node
    input_data : pd.Dataframe
        Dataframe the user wants to find labels for rows
    
    ...

    Returns
    -------
    String
        row predicted label
    """
    if node.label is not None:
        return node.label
    else:
        if isinstance(node.threshold, str):
            # Comparison if this node splits on a string value
            if input_data.loc[node.feature] == node.threshold:
                return predict(node.left, input_data)
            else:
                return predict(node.right, input_data)
        else:
            # Comparison if this node splits on a number value
            if input_data.loc[node.feature] <= node.threshold:
                return predict(node.left, input_data)
            else:
                return predict(node.right, input_data)
    

def best_split_df(dataframe: pd.DataFrame, labels, is_numerical: pd.Series):
    """
    Finding the best split looping over an entire dataframe

    ...

    Paramters
    ---------
    dataframe : pd.Dataframe
        Dataframe used fro training the tree
    labels : pd.Series
        Series of labels for training data that gives each row a classification
    is_numerical : pd.Series(list, list)
        Series of a tuple of two lists. The first list is a list of booleans telling if the column is numerical or categorical.
        The second list in the tuple are the column names except for the last (the labels column)
        
    ...

    Returns
    -------
    tuple -- (string, string/float/int)
        best column and threshold within that column to split
    """
    best_threshold = None
    col = None
    best_impurity = np.inf
    # iteratre through entire dataframe to find best split
    for col_name in dataframe:
        threshold, impurity = best_split_col(dataframe[col_name], labels, is_numerical[col_name])
        # saving lowest impurity
        if impurity < best_impurity:
            best_impurity, best_threshold, col = impurity, threshold, col_name

    return col, best_threshold

def best_split_col(data: pd.Series, labels: pd.Series, is_numerical: bool):
    """
    Finding the best split in a column

    ...

    Paramters
    ---------
    dataframe : pd.Dataframe
        Dataframe used fro training the tree
    labels : pd.Series
        Series of labels for training data that gives each row a classification
    is_numerical : pd.Series(list, list)
        Series of a tuple of two lists. The first list is a list of booleans telling if the column is numerical or categorical.
        The second list in the tuple are the column names except for the last (the labels column)
        
    ...

    Returns
    -------
    tuple -- float/int/string, float
        best threshold value in the column and the impurity of the split
    """
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

        weighted_impurity = weighted_gini_impurity(left_labels, right_labels)
        
        # save the best weighted impurity and threshold
        if weighted_impurity < impurity:
            impurity = weighted_impurity
            threshold = value
    return threshold, impurity

def split(dataframe: pd.DataFrame, col_name, threshold, labels, is_numerical):
    """
    Splitting the dataframe at the given column and threshold

    ...

    Paramters
    ---------
    dataframe : pd.Dataframe
        Dataframe being split
    col_name : pd.Series
        name of the column used for the split
    threshold : Float, int, or string
        threshold value split at
    labels : pd.Series
        Series of labels for training data that gives each row a classification
    is_numerical : pd.Series(list, list)
        Series of a tuple of two lists. The first list is a list of booleans telling if the column is numerical or categorical.
        The second list in the tuple are the column names except for the last (the labels column)
        
    ...

    Returns
    -------
    pd.Dataframe
        two split dataframes that match the threshold or do not and splits the labels series accordinly 
    """
    col_names = list(dataframe.columns)
    idx = col_names.index(col_name)
    # splitting the data and labels based on the column and if it matches the threshold
    if is_numerical[idx]:
        return dataframe[dataframe[col_name] <= threshold], dataframe[dataframe[col_name] > threshold], \
            labels[dataframe[col_name] <= threshold], labels[dataframe[col_name] > threshold]
    return dataframe[dataframe[col_name] == threshold], dataframe[dataframe[col_name] != threshold], \
        labels[dataframe[col_name] == threshold], labels[dataframe[col_name] != threshold]

def gini_imp(classes):
    """
    Calculating the gini impurity for a given set of data

    ...

    Paramters
    ---------
    classes : pd.Series
        Series of data used to calculate the gini impurity. Contains only classes since we only need classes
        when calculating the gini impurity of a set of data.

    ...

    Returns
    -------
    float
        gini impurity
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
    Calculating the weighted gini impurity of the left and right node split

    ...

    Paramters
    ---------
    left : Node
        left gini impurity
    right : Node
        right gini impurity

    ...

    Returns
    -------
    float
        weighted gini impurity
    """
    #gini from left node
    left_gini = gini_imp(left)

    #gini from right node
    right_gini = gini_imp(right)

    # weighted gini impurity
    weighted_gini = (left_gini * (len(left) / (len(left) + (len(right))))) + \
                    (right_gini * (len(right) / (len(left) + (len(right)))))

    return weighted_gini


def tree_score(tree, df, labels):
    """
    Measures the accuracy of the decision tree

    ...

    Paramters
    ---------
    tree : DecisionTree
        Decision Tree that was built to classify data
    df : pd.Dataframe
        Dataframe used to build the decision tree
    labels : pd.Series
        the column of labels used to classify

    ...

    Returns
    -------
    float
        the accuracy of the tree
    """
    
    count = 0

    # iterating through df
    for i in range(len(df)):

        # calls predict() function on each row
        prediction = predict(tree, df.iloc[i])

        # check if prediction is correct, and increase count by 1 if it is
        if prediction == labels.iloc[i]:
            count += 1

    # returns the accuracy measurement
    return count / len(df)


def gen_is_numerical(df):
    """ 
    Generates an 'is-numerical' series based on the dataframe given.

    An 'is-numerical' series is a series that has a True or False value for whether a given
    column is numeric or not.
    
    ...

    Paramters
    ---------
    df : pd.Dataframe
        training dataframe given

    ...

    Returns
    -------
    pd.Series
        tuple with a list of booleans and the column names without the labels column
    """

    bool_list = []
    
    # get first row to see what data types we have, but could've picked any row
    first_row = df.iloc[0]

    # iterate through the columns of the first row
    for i in range(len(first_row)):
        # checking if the columns are numeric or a string and put the
        # correspoding boolean into a list
        if isinstance(first_row.iloc[i], str):
            bool_list.append(False)
        else:
            bool_list.append(True)

    # returns the series with the data type of the columns and ignores last column
    return pd.Series(bool_list, df.columns[:len(df.columns)])


def train_test_split(df, train_size, random_seed=None):
    """
    Returns a training dataframe, a test data dataframe, and a test classes dataframes.
    This is used for randomly splitting the training and testing dataset in order achieve
    good results.

    ...
    
    Attributes
    ----------
    train_size : Float
        Percent of rows that will used to train the decision tree
    random_seed : Int
        Randomly generated number that will be the row that will be chosen
        to get trained
    
    ...

    Returns
    -------
    Dataframe, Dataframe, Series
        Xy_train, which is the classes and data combined in a dataset, x_test,
        which are the node features, and y_test, which are the node labels
    """

    if random_seed is not None:
        random.seed(random_seed)

    # First, get the number of rows we want for our training set
    num_train_rows = int(len(df) * train_size)

    # Put all the possible indices of the dataframe in a list. This list helps keep
    # track of which indices are used and which ones are left.
    possible_idxs = [idx for idx in range(len(df))]

    Xy_train = pd.DataFrame(columns=df.columns)
    Xy_test = pd.DataFrame(columns=df.columns)

    for idx in range(num_train_rows):
        # Generate a random index for the possible_idxs list
        rand_idx = random.randint(0, len(possible_idxs) - 1)

        # Use that random index for possible_idxs to get a random index of the dataframe
        Xy_train.loc[idx] = df.iloc[possible_idxs[rand_idx]]

        # Remove the element at that index from the list since it has been used
        del possible_idxs[rand_idx]

    # Now, let's fill up the Xy_test dataframe by zipping the index for the Xy_test
    # dataframe and the remaining indexes of the df dataframe.
    for test_idx, df_idx in zip(range(len(possible_idxs)), possible_idxs):
        Xy_test.loc[test_idx] = df.iloc[df_idx]

    # Drop the classes column
    X_test = Xy_test.drop(Xy_test.columns[-1], axis=1)

    # Extract only the class column as a Series
    y_test = Xy_test[Xy_test.columns[-1]]

    return Xy_train, X_test, y_test    
 
