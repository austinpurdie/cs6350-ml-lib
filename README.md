# cs6350-ml-lib

This is a machine learning library developed by Austin Purdie for CS5350/6350 in University of Utah.

# DecisionTree

The DecisionTree folder contains code that can be used to learn decision trees. The code that's used to build the tree is in the DecisionTree.py script. To learn a new tree, you can run 

```python

    tree = DecisionTree(max_depth)
    build_tree(tree, data, target, parent, method)

```

where the parameters are as follows:

- max_depth: the maximum depth of the tree. When this argument is 0, the tree will be built with no maximum depth limitations.

- tree: a DecisionTree() object. The user should pass an empty tree to build_tree as shown in the code snippet above.

- data: a pandas data frame containing the data set to build the tree with. 

- target: the column name of the data set's target or label.

- parent: should always be tree.root.

- method: the method to calculate information gain; valid arguments are 'entropy', 'majority error', and 'gini index'.
