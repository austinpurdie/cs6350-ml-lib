import DecisionTree as dt
import pandas as pd
import math
import numpy as np
import sys

print("\nAdaBoost Implementation")
sys.stdout.flush()
print("\nGetting data, setting parameters...")
sys.stdout.flush()

iterations = 500

colnames = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']

numeric_cols = [0, 5, 9, 11, 12, 13, 14]

train_data = dt.numeric_to_binary(pd.read_csv("EnsembleLearning/Data/bank-train.csv", names = colnames, header = None), numeric_cols)
train_data['y'] = dt.adaboost_target_encoder(list(train_data['y']), 'yes', 'no')
train_actual = np.array(train_data['y'])


test_data = dt.numeric_to_binary(pd.read_csv("EnsembleLearning/Data/bank-test.csv", names = colnames, header = None), numeric_cols)
test_data['y'] = dt.adaboost_target_encoder(list(test_data['y']), 'yes', 'no')
test_actual = np.array(test_data['y'])

adaboost_objects = dt.build_decision_tree_adaboost_model(train_data, 'y', 'entropy', iterations, depth = 1)

tree_list = adaboost_objects[0]
alpha_list = adaboost_objects[1]

print("\nGenerating test predictions...")
sys.stdout.flush()
adaboost_test_accuracy = dt.get_adaboost_accuracy(tree_list, alpha_list, test_data, 'y')
test_stump_accuracy = []
for i in tree_list:
    temp_pred = dt.get_tree_predictions(i, test_data)
    test_stump_accuracy.append(dt.get_test_accuracy(test_data, 'y', temp_pred))

print("\nGenerating train predictions...")
sys.stdout.flush()
adaboost_train_accuracy = dt.get_adaboost_accuracy(tree_list, alpha_list, train_data, 'y')
train_stump_accuracy = []
for i in tree_list:
    temp_pred = dt.get_tree_predictions(i, train_data)
    train_stump_accuracy.append(dt.get_test_accuracy(train_data, 'y', temp_pred))

iters = []
for j in range(len(tree_list)):
    iters.append(j + 1)

adaboost_accuracy_dict = {'Iteration': iters, 'AdaBoost Test Accuracy': adaboost_test_accuracy['Accuracy'], 'AdaBoost Train Accuracy': adaboost_train_accuracy['Accuracy'], 'Stump Test Accuracy': test_stump_accuracy, 'Stump Train Accuracy': train_stump_accuracy}
adaboost_accuracy_df = pd.DataFrame.from_dict(adaboost_accuracy_dict, orient = 'columns')

adaboost_accuracy_df.to_csv('adaboost-accuracy.csv')
