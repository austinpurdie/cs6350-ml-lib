import DecisionTree as dt
import pandas as pd
import math
import numpy as np

print("\nAdaBoost Implementation")
print("\nGetting data, setting parameters...")

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
adaboost_test_accuracy = dt.get_adaboost_accuracy(tree_list, alpha_list, test_data, 'y')
print("\nGenerating train predictions...")
adaboost_train_accuracy = dt.get_adaboost_accuracy(tree_list, alpha_list, train_data, 'y')

iters = []
for j in range(len(tree_list)):
    iters.append(j + 1)

adaboost_accuracy_dict = {'Iteration': iters, 'Test Accuracy': adaboost_test_accuracy['Accuracy'], 'Train Accuracy': adaboost_train_accuracy['Accuracy']}
adaboost_accuracy_df = pd.DataFrame.from_dict(adaboost_accuracy_dict, orient = 'columns')

adaboost_accuracy_df.to_csv('adaboost-accuracy.csv')