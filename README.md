# Simple Decision Tree Classfier using Gini Impurity
The purpose of this simple decision tree classifier is to make using a decision tree easier for people new to machine learning and for those who are not in a data-science type discipline.
Additionally, the file has a single end goal, which is to build a decision tree because of this everything is organized in one file. 
it should be easy for someone to input their CSV data as seen below to do their classification predictions in whatever setting.
The main objects are at the top, the Node and the Decision Tree classes. 
In those functions the necessary dataframe looping, gini impurity calculations, dataframe splitting, train and test, tree score, and is numerical were implemented to achieve a clean decision tree class.
The decision tree class is the most important, thus it was kept clean and easily readable calling the helper functions that can be seen below. 
```python
    columns = ["sex", "length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight", "rings"]

    df = pd.read_csv("iris.csv")

    # df.columns = columns

    Xy_train, X_test, y_test = train_test_split(df=df, train_size=0.50)

    print("Xy_train = \n", Xy_train)
    print("X_test = \n", X_test)
    print("y_test = \n", y_test)

    tree = DecisionTree(Xy_train)
    node = tree.train()
    print(node)

    print(tree_score(node, X_test, y_test))
```
