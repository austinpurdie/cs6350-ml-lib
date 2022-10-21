import pandas as pd
import numpy as np
import random
import DecisionTree as dt
import sys

iterations = 500
size = 5000

numeric_cols = [4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
colnames = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'DEFAULT_IND']

train_data = pd.read_csv('EnsembleLearning/Data/default-train.csv')
test_data = pd.read_csv('EnsembleLearning/Data/default-test.csv')

train_data['DEFAULT_IND'] = dt.adaboost_target_encoder(train_data['DEFAULT_IND'], 1, 0)
test_data['DEFAULT_IND'] = dt.adaboost_target_encoder(train_data['DEFAULT_IND'], 1, 0)

train_actual = train_data['DEFAULT_IND']
test_actual = test_data['DEFAULT_IND']

#####################
# BEGIN SINGLE TREE #
#####################

print("\n(1/10) Building single tree...")
sys.stdout.flush()
single_tree = dt.DecisionTree(0)
dummy_weight_train_data = train_data
dummy_weight_train_data['weights'] = [1/len(train_data)] * len(train_data)
dt.build_tree(single_tree, dummy_weight_train_data, 'DEFAULT_IND', 'weights', single_tree.root, 'entropy')

single_train_predictions = dt.get_tree_predictions(single_tree, train_data)
single_test_predictions = dt.get_tree_predictions(single_tree, test_data)

single_train_accuracy = dt.get_test_accuracy(train_data, 'DEFAULT_IND', single_train_predictions)
single_test_accuracy = dt.get_test_accuracy(test_data, 'DEFAULT_IND', single_test_predictions)

single_accuracy = np.array([single_train_accuracy, single_test_accuracy])
np.savetxt("default-single-accuracy.csv", single_accuracy, delimiter = ",")


##################
# BEGIN ADABOOST #
##################

print("\n(2/10) Building AdaBoost stumps...")
sys.stdout.flush()
adaboost_objects = dt.build_decision_tree_adaboost_model(train_data, 'DEFAULT_IND', 'entropy', iterations, depth = 1)

tree_list = adaboost_objects[0]
alpha_list = adaboost_objects[1]

print("\n(3/10) Generating AdaBoost test predictions...")
sys.stdout.flush()
sys.stdout.flush()
adaboost_test_accuracy = dt.get_adaboost_accuracy(tree_list, alpha_list, test_data, 'DEFAULT_IND')
test_stump_accuracy = []
for i in tree_list:
    temp_pred = dt.get_tree_predictions(i, test_data)
    test_stump_accuracy.append(dt.get_test_accuracy(test_data, 'DEFAULT_IND', temp_pred))

print("\n(4/10)Generating AdaBoost train predictions...")
sys.stdout.flush()
adaboost_train_accuracy = dt.get_adaboost_accuracy(tree_list, alpha_list, train_data, 'DEFAULT_IND')
train_stump_accuracy = []
for i in tree_list:
    temp_pred = dt.get_tree_predictions(i, train_data)
    train_stump_accuracy.append(dt.get_test_accuracy(train_data, 'DEFAULT_IND', temp_pred))

iters = []
for j in range(len(tree_list)):
    iters.append(j + 1)

adaboost_accuracy_dict = {'Iteration': iters, 'AdaBoost Test Accuracy': adaboost_test_accuracy['Accuracy'], 'AdaBoost Train Accuracy': adaboost_train_accuracy['Accuracy'], 'Stump Test Accuracy': test_stump_accuracy, 'Stump Train Accuracy': train_stump_accuracy}
adaboost_accuracy_df = pd.DataFrame.from_dict(adaboost_accuracy_dict, orient = 'columns')

adaboost_accuracy_df.to_csv('default-adaboost-accuracy.csv')

#################
# BEGIN BAGGING #
#################

print("(5/10) Building trees for bagging...")
sys.stdout.flush()
bagged_trees = dt.build_bagged_decision_tree_model(train_data, 'DEFAULT_IND', 'entropy', num_iterations = iterations, bag_size = size)

test_actual = list(test_data['DEFAULT_IND'])
train_actual = list(train_data['DEFAULT_IND'])

print("(6/10) Generating bagging test predictions and accuracies...")
sys.stdout.flush()
test_accuracy = dt.get_bagged_accuracy(bagged_trees, test_data, 'DEFAULT_IND')

print("(7/10) Generating bagging train predictions and accuracies...")
sys.stdout.flush()
train_accuracy = dt.get_bagged_accuracy(bagged_trees, train_data, 'DEFAULT_IND')

iters = []
for j in range(iterations):
    iters.append(j + 1)

accuracy_dict = {'Iteration': iters, 
                    'Test': test_accuracy.loc[:, 'Accuracy'],
                    'Train': train_accuracy.loc[:, 'Accuracy']}

final_accuracy_df = pd.DataFrame.from_dict(accuracy_dict, orient = 'columns')

final_accuracy_df.to_csv('default-bagging-accuracy.csv')

#######################
# BEGIN RANDOM FOREST #
#######################

print("(8/10) Building trees for random forest...")
sys.stdout.flush()
rf_trees = dt.build_bagged_decision_tree_model(train_data, 'DEFAULT_IND', 'entropy', num_iterations = iterations, bag_size = size, rand_flag = 2)

test_actual = list(test_data['DEFAULT_IND'])
train_actual = list(train_data['DEFAULT_IND'])

print("(9/10) Generating random forest test predictions and accuracies...")
sys.stdout.flush()
test_accuracy = dt.get_bagged_accuracy(rf_trees, test_data, 'DEFAULT_IND')

print("(10/10) Generating random forest train predictions and accuracies...")
sys.stdout.flush()
train_accuracy = dt.get_bagged_accuracy(rf_trees, train_data, 'DEFAULT_IND')

iters = []
for j in range(iterations):
    iters.append(j + 1)

accuracy_dict = {'Iteration': iters, 
                    'Test': test_accuracy.loc[:, 'Accuracy'],
                    'Train': train_accuracy.loc[:, 'Accuracy']}

final_accuracy_df = pd.DataFrame.from_dict(accuracy_dict, orient = 'columns')

final_accuracy_df.to_csv('default-random-forest-accuracy.csv')

