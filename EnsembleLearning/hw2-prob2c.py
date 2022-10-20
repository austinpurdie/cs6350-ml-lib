import pandas as pd
import numpy as np
import random
import DecisionTree as dt

print("\nGetting data, setting parameters...")

colnames = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']

numeric_cols = [0, 5, 9, 11, 12, 13, 14]

train_data = dt.numeric_to_binary(pd.read_csv("EnsembleLearning/Data/bank-train.csv", names = colnames, header = None), numeric_cols)
train_data['y'] = dt.adaboost_target_encoder(train_data['y'], 'yes', 'no')
train_actual = np.array(train_data['y'], dtype = int)

test_data = dt.numeric_to_binary(pd.read_csv("EnsembleLearning/Data/bank-test.csv", names = colnames, header = None), numeric_cols)
test_data['y'] = dt.adaboost_target_encoder(test_data['y'], 'yes', 'no')
test_actual = np.array(test_data['y'], dtype = int)

outer_iterations = 10
inner_iterations = 10
outer_size = 1000
inner_size = 250

bagged_predictors = []

for i in range(outer_iterations):
    print("Building a lot of trees, this will take a while... " + f"{(i+1)/outer_iterations*100:.1f} %", end="\r")
    outer_sample_indices = random.sample(range(len(train_data)), k = outer_size)
    outer_sample_list = []
    for j in outer_sample_indices:
        new_entry = list(train_data.loc[j, :])
        outer_sample_list.append(new_entry)
    outer_sample = pd.DataFrame(outer_sample_list, columns = train_data.columns)
    bagged_trees = dt.build_bagged_decision_tree_model(outer_sample, 'y', 'entropy', num_iterations = inner_iterations, bag_size = inner_size, verbose = False)
    bagged_predictors.append(bagged_trees)

first_trees = []
for j in range(len(bagged_predictors)):
    first_trees.append(bagged_predictors[j][0])

first_tree_predictions = []
for k in first_trees:
    first_tree_predictions.append(dt.get_tree_predictions(k, test_data))

first_tree_predictions = np.array(first_tree_predictions)
first_tree_means = np.mean(first_tree_predictions, axis = 0)

first_tree_bias = np.mean((first_tree_means - test_actual)**2)
first_tree_variance = np.mean(np.var(first_tree_predictions, axis = 0))
first_tree_gse = first_tree_bias + first_tree_variance

print("\nSingle Tree GSE: " + str(first_tree_gse))
print("\nSingle Tree Variance: " + str(first_tree_variance))
print("\nSingle Tree Bias: " + str(first_tree_bias))

bagged_predictions = []

for l in range(len(bagged_predictors)):
    bagged_predictions.append(dt.get_bagged_accuracy(bagged_predictors[l], test_data, 'y', iter_accuracies = False)[1])

bagged_predictions = np.array(bagged_predictions)
bagged_means = np.mean(bagged_predictions, axis = 0)

bagged_bias = np.mean((bagged_means - test_actual)**2)
bagged_variance = np.mean(np.var(bagged_predictions, axis = 0))
bagged_gse = bagged_bias + bagged_variance

print("\nBagged Trees GSE: " + str(bagged_gse))
print('\nBagged Tree Variance: ' + str(bagged_variance))
print('\nBagged Tree Bias: ' + str(bagged_bias))


