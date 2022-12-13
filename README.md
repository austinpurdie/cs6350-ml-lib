# cs6350-ml-lib

This is a machine learning library developed by Austin Purdie for CS5350/6350 in University of Utah.

# A Note to the Grader

To run the script associated with homework #3, run the run-hw3.sh script.

To run the scripts associated with homework #2, run the run-hw2.sh script. For this assignment, I chose to produce my visualizations in R. You can see these visualizations in the PDF submission I attached via Canvas, but the bash script will not reproduce them. You may view the script I used to generate them in the main parent directory of the repository.

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

# Perceptron

The Perceptron folder contains scripts that can run the standard, voted, and averaged perceptron algorithms. You can run 

```python

perceptron(data, target, r, epochs, type)

```

to run any of the three versions of the perceptron algorithm. The parameters are as follows:

- data: the dataset you wish to train with; all of the data must be numerical

- target: the name of the target column, which must be composed only of 1's and -1's

- r: the learning rate (try starting with 1)

- epochs: the number of times to run the algorithm through the data set

- type: a string indicating which algorithm type to use; valid arguments are 'standard' and 'voted'. There is no 'average' argument because the average perceptron only differs from the voted perceptron in the way it generates predictions, which is a task performed by the get_perceptron_accuracy() function.

You can test your model's accuracy by calling the get_perceptron_accuracy() function:

```python

get_perceptron_accuracy(data, target, w, c, type)

```

where the parameters are as follows:

- data: the data you wish to generate predictions for 

- target: the name of the target column

- w: the weight vector returned by perceptron() above

- c: the vector of votes returned by perceptron() above, if you wish to get voted accuracy and the perceptron() function's type argument was 'voted'

- type: a string indicating which algorithm type to use; if perceptron() was run with type = 'standard', then this same argument should be passed to the accuracy function. If perceptron() was run with type = 'voted', then this argument can be 'voted' or 'average'.

## Support Vector Machines

You can run

```python

build_svm(data, target, num_epochs, c, gamma_0, a = False)

```

to run the stochastic gradient descent algorithm. The parameters are as follows:

- data: the training data set

- target: the array of example labels; these should all be 1 or -1

- num_epochs: the number of times to run through the data set

- c: the parameter C used to compute each new iteration of w

- gamma_0: the learning rate parameter

- a: another learning rate parameter; leave as false to use the schedule from part (b) of Problem 2, otherwise replace this with a number to use the schedule from part (a)

## Neural Networks

You can run

```python

net = NeuralNetwork(architecure, initialize)

```

to initialize a new neural network, where

- architecture: a list denoting the network's architecture; for example, [4, 3, 3, 1] would be a network with input size 4, two hidden layers with three neurons, and one output node

- initialize: a string indicating how to initialize the network weights; 'gaussian' and 'zero' are valid arguments

From there, the network can be trained via stochastic gradient descent by running

```python

net.stoch_grad(data, labels, epochs, gamma_0, d)

```

where the parameters are as follows:

- data: a NumPy array of the examples; do not include labels

- labels: a NumPy array of the labels corresponding the examples

- epochs: the number of times to cycle through the data in the stochastic gradient descent algorithm

- gamma_0: a learning rate parameter; try 1 to start

- d: a learning rate parameter; try 50 to start

Once the network has been trained, you can feed a test set to get a classification accuracy rate by running

```python

net.get_accuracy(examples, labels)

```

where the parameters are as they were in net.stoch_grad() above.