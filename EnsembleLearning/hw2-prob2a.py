import DecisionTree as dt
import pandas as pd
import math

report_dict = {}

colnames = ['outlook', 'temperature', 'humidity', 'wind', 'play']

# colnames = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']

numeric_cols = [0, 5, 9, 11, 12, 13, 14]

train_data = pd.read_csv("EnsembleLearning/DELETE-tennis_test.csv", names = colnames, header = None)
train_data['weights'] = [1/len(train_data)] * len(train_data)

labels_to_be_encoded = list(train_data['play'])
for m in range(len(labels_to_be_encoded)):
    if labels_to_be_encoded[m] == 'Y':
        labels_to_be_encoded[m] = 1
    else:
        labels_to_be_encoded[m] = -1

train_data['play'] = labels_to_be_encoded

alpha_list = []
trees = []

for i in range(10):
    temp = dt.DecisionTree(1)
    dt.build_tree(temp, train_data, 'play', 'weights', temp.root, 'entropy')
    trees.append(temp)
    predictions = dt.get_tree_predictions(temp, train_data)
    actual = list(train_data['play'])
    correct = []
    accuracy_df = pd.DataFrame(list(zip(actual, predictions)), columns = ['actual', 'predictions'])
    num_correct = 0
    for k in range(len(accuracy_df)):
        if accuracy_df.loc[k, 'actual'] == accuracy_df.loc[k, 'predictions']:
            num_correct += 1
            correct.append(1)
        else:
            correct.append(-1)
    accuracy = num_correct / len(accuracy_df)
    error = 1 - accuracy
    alpha = 0.5 * math.log((1 - error)/error)
    alpha_list.append(alpha)
    new_weights = list(train_data['weights'])
    for j in range(len(new_weights)):
        new_weights[j] = new_weights[j] * math.exp(-alpha * train_data.loc[j, 'play'] * correct[j])
    z = sum(new_weights)
    new_weights = [x / z for x in new_weights]
    train_data['weights'] = new_weights


# train_data = dt.numeric_to_binary(pd.read_csv("EnsembleLearning/Data/bank-train.csv", names = colnames, header = None), numeric_cols)
# train_data['weights'] = [1/len(train_data)] * len(train_data)

# test_data = dt.numeric_to_binary(pd.read_csv("EnsembleLearning/Data/bank-test.csv", names = colnames, header = None), numeric_cols)


# for i in range(0, 500):
