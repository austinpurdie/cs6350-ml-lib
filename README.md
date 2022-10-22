# cs6350-ml-lib

This is a machine learning library developed by Austin Purdie for CS5350/6350 in University of Utah.

# A Note to the Grader

To run the scripts associated with homework #2, run the run-hw2.sh script. For this assignment, I chose to produce my visualizations in R. The bash script will not reproduce these visualizations, but the script I used to generate them can be viewed in the main parent directory of the repository.

To run the scripts associated with homework #1, run the run-hw1.sh script.


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


# EnsembleLearning

The EnsembleLearning folder contains a modified version of the DecisionTree.py module that adds support for the AdaBoost, bagging, and random forest algorithms. 

## AdaBoost

You can build an AdaBoost model by running 

```python

adaboost_objects = build_decision_tree_adaboost_model(data, target, method, num_iterations)

```

where the parameters are as follows:

- data: the training data set, either in the form of a pandas dataframe or numpy array

- target: the name of the target column name in data

- method: instructions on how to calculate information gain; 'entropy', 'majority_error', or 'gini_index'

- num_ierations: the number of weak classifiers to build

Once the model is developed, you can call 

```python

adaboost_objects[0]
adaboost_objects[1]

```

to access a list of the weak classifers and their votes respectively. To get predictions on a test data set, you can run


```python

get_adaboost_accuracy(adaboost_objects[0], adaboost_objects[1], data, target)

```

where data is the test set and target is the name of the target column name in the data.

## Bagging and Random Forest

You can build a bagged decision tree model by running

```python

build_bagged_decision_tree_model(data, target, method, num_iterations, bag_size, rand_flag)

```

where the parameters are as follows:

- data: the training data set, either in the form of a pandas dataframe or numpy array

- target: the name of the target column name in data

- method: instructions on how to calculate information gain; 'entropy', 'majority_error', or 'gini_index'

- num_ierations: the number of trees to build

- bag_size: the sample size to be selected from the training data for building each tree

- rand_flag: the default is False, but you can build a random forest model instead by passing rand_flag = True

To get predictions on a test data set, you can run

```python

get_bagged_accuracy(trees, data, target, iter_accuracies)

```

where the parameters are as follows:

- trees: the list of trees produced during the construction of the model, which is returned by build_bagged_decision_tree_model()

- data: the test data set

- target: the target column name

- iter_accuracies: this is True by default, which causes the test accuracy to be recorded for each iteration; pass iter_accuracies = False if you're only interested in the final test accuracy

The function returns a list containing two more lists; one is the list of test accuracies and the other is the list of test predictions.

# LinearRegression

The LinearRegression folder contains a script to run the batch and stochastic gradient descent algorithms. 

## Batch Gradient Descent

You can run 

```python

backtracking_gradient(w0, x, y, epsilon, max_iter, s, alpha, beta)

```

to run the batch gradient descent algorithm with backtracking line search. The backtracking line search updates the learning rate at each iteration rather than using a constant step size. The s, alpha, and beta parameters are what dictate how the learning rate is updated. It's recommended to begin with s = 1, alpha = 0.5, and beta = 0.5 and then tune from there as needed.

- w0: the vector to initialize the algorithm on

- x: the training data set, not including the target

- y: the vector of actual target values

- epsilon: the convergence threshold; the algorithm will stop when the norm of the gradient is less than this parameter

- max_iter: the maximum iterations; the algorithm will stop once it iterates a number of times equal to this parameter, regardless of whether the convergence criterion has been met

- s: the initial guess for the learning rate, to be tuned by alpha and beta

- alpha: a tuning parameter for the backtracking line search

- beta: a tuning parameter for the backtracking line search

The function will return a list containing 1) the learned weight vector w, 2) the value of the cost function at that weight vector, 3) the number of iterations, 4) the list of learning rates used at each iteration, and 5) the list of values of the cost function at each iteration.

## Stochastic Gradient Descent

You can run

```python

stochastic_gradient(w0, x, y, r, d, epsilon, max_iter)

```

to run the stochastic gradient descent algorithm. The parameters are as follows:

- w0: the vector to initialize the algorithm on

- x: the training data set, not including the target

- y: the vector of actual target values

- r: the initial learning rate

- d: a factor to reduce the learning rate by after each pass through the entire data set; for constant learning rate, set d = 1. Otherwise, start with a number very close to one (e.g., d = 0.9999) and tune from there. Rates greater than 1 are not recommended as the learning rate will increase which will prevent the algorithm from converging.

- epsilon: the convergence threshold; the algorithm will stop when the norm of the gradient is less than this parameter

- max_iter: the maximum iterations; the algorithm will stop once it iterates a number of times equal to this parameter, regardless of whether the convergence criterion has been met

The function will return a list containing 1) the learned weight vector w, 2) the value of the cost function at that weight vector, 3) the number of iterations, and 4) the list of values of the cost function at each iteration.

