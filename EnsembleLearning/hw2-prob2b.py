import pandas as pd
import numpy as np
import DecisionTree as dt
import sys

print("Bagging Implementation")
sys.stdout.flush()
print("Getting data, setting parameters...")
sys.stdout.flush()

colnames = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']

numeric_cols = [0, 5, 9, 11, 12, 13, 14]

train_data = dt.numeric_to_binary(pd.read_csv("EnsembleLearning/Data/bank-train.csv", names = colnames, header = None), numeric_cols)

test_data = dt.numeric_to_binary(pd.read_csv("EnsembleLearning/Data/bank-test.csv", names = colnames, header = None), numeric_cols)

iterations = 500
size = 5000

print("(1/3) Building trees for bagging...")
sys.stdout.flush()
bagged_trees = dt.build_bagged_decision_tree_model(train_data, 'y', 'entropy', num_iterations = iterations, bag_size = size)

test_actual = list(test_data['y'])
train_actual = list(train_data['y'])

print("(2/3) Generating bagging test predictions and accuracies...")
sys.stdout.flush()
test_accuracy = dt.get_bagged_accuracy(bagged_trees, test_data, 'y')

print("(3/3) Generating bagging train predictions and accuracies...")
sys.stdout.flush()
train_accuracy = dt.get_bagged_accuracy(bagged_trees, train_data, 'y')

iters = []
for j in range(iterations):
    iters.append(j + 1)

accuracy_dict = {'Iteration': iters, 
                    'Test': test_accuracy.loc[:, 'Accuracy'],
                    'Train': train_accuracy.loc[:, 'Accuracy']}

final_accuracy_df = pd.DataFrame.from_dict(accuracy_dict, orient = 'columns')

final_accuracy_df.to_csv('bagging-accuracy.csv')



