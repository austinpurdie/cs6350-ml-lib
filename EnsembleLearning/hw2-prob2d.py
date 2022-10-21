import pandas as pd
import numpy as np
import DecisionTree as dt
import sys

print("Random Forest Implementation")
sys.stdout.flush()
print("Getting data, setting parameters...")
sys.stdout.flush()

colnames = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']

numeric_cols = [0, 5, 9, 11, 12, 13, 14]

train_data = dt.numeric_to_binary(pd.read_csv("EnsembleLearning/Data/bank-train.csv", names = colnames, header = None), numeric_cols)

test_data = dt.numeric_to_binary(pd.read_csv("EnsembleLearning/Data/bank-test.csv", names = colnames, header = None), numeric_cols)

iterations = 500
size = 750

print("(1/9) Building trees for random forest (k = 2)...")
sys.stdout.flush()
bagged_trees_2 = dt.build_bagged_decision_tree_model(train_data, 'y', 'entropy', num_iterations = iterations, bag_size = size, rand_flag = 2)

print("(2/9) Building trees for random forest (k = 4)...")
sys.stdout.flush()
bagged_trees_4 = dt.build_bagged_decision_tree_model(train_data, 'y', 'entropy', num_iterations = iterations, bag_size = size, rand_flag = 4)

print("(3/9) Building trees for random forest (k = 6)...")
sys.stdout.flush()
bagged_trees_6 = dt.build_bagged_decision_tree_model(train_data, 'y', 'entropy', num_iterations = iterations, bag_size = size, rand_flag = 6)

test_actual = list(test_data['y'])
train_actual = list(train_data['y'])

print("(4/9) Generating random forest (k = 2) test predictions and accuracies...")
sys.stdout.flush()
test_accuracy_2 = dt.get_bagged_accuracy(bagged_trees_2, test_data, 'y')

print("(5/9) Generating random forest (k = 2) train predictions and accuracies...")
sys.stdout.flush()
train_accuracy_2 = dt.get_bagged_accuracy(bagged_trees_2, test_data, 'y')

print("(6/9) Generating random forest (k = 4) test predictions and accuracies...")
sys.stdout.flush()
test_accuracy_4 = dt.get_bagged_accuracy(bagged_trees_4, test_data, 'y')

print("(7/9) Generating random forest (k = 4) train predictions and accuracies...")
sys.stdout.flush()
train_accuracy_4 = dt.get_bagged_accuracy(bagged_trees_4, train_data, 'y')

print("(8/9) Generating random forest (k = 6) test predictions and accuracies...")
sys.stdout.flush()
test_accuracy_6 = dt.get_bagged_accuracy(bagged_trees_6, test_data, 'y')

print("(9/9) Generating random forest (k = 6) train predictions and accuracies...")
sys.stdout.flush()
train_accuracy_6 = dt.get_bagged_accuracy(bagged_trees_6, train_data, 'y')

iters = []
for j in range(iterations):
    iters.append(j + 1)

accuracy_dict = {'Iteration': iters, 
                    'k = 2, Test': test_accuracy_2.loc[:, 'Accuracy'],
                    'k = 2, Train': train_accuracy_2.loc[:, 'Accuracy'],
                    'k = 4, Test': test_accuracy_4.loc[:, 'Accuracy'],
                    'k = 4, Train': train_accuracy_4.loc[:, 'Accuracy'],
                    'k = 6, Test': test_accuracy_6.loc[:, 'Accuracy'],
                    'k = 6, Train': train_accuracy_6.loc[:, 'Accuracy']}

final_accuracy_df = pd.DataFrame.from_dict(accuracy_dict, orient = 'columns')

final_accuracy_df.to_csv('random-forest-accuracy.csv')

