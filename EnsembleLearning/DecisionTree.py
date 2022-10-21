import math
import pandas as pd
import numpy as np
import random
import sys

pd.options.mode.chained_assignment = None

class DecisionTreeNode:
    def __init__(self, attribute, value, type, parent):
        self.attribute = attribute # which attribute the node is giving instructions for (e.g., outlook, humidity)
        self.value = value # the value of the attribute the node is giving instructions for (e.g., the outlook is //sunny//)
        self.type = type # whether the node is a root, branch, or leaf
        self.parent = parent # the node's parent node
        self.child = [] # a list of nodes that this node points to in the tree
        self.depth = 0 # the depth of the tree is maintained at the level of each node for convenience of access
        self.most_common = None # the most common label at each particular node is recorded

class DecisionTree:
    def __init__(self, max_depth, rand_flag = False):
        self.root = DecisionTreeNode(None, None, "root", None)
        self.root.rand_flag = rand_flag
        self.max_depth = max_depth
        self.rand_flag = rand_flag
        self.node_count = 0

    def add_branch(self, parent, attribute, value):
        new_node = DecisionTreeNode(attribute, value, "branch", parent)
        new_node.parent.child.append(new_node)
        new_node.depth = new_node.parent.depth + 1
        self.node_count += 1
        return new_node

    def add_leaf(self, parent, value):
        new_leaf = DecisionTreeNode(None, value, 'leaf', parent)
        new_leaf.parent.child.append(new_leaf)

def read_data(data, features):
    df = pd.read_csv(data, names = features)
    return df

def label_proportions(labels_weights):
    labels_weights.columns = ['labels', 'weights']
    weight_list = list(labels_weights['weights'])
    weight_sum = sum(weight_list)
    standardized_weights = [x / weight_sum for x in weight_list]
    labels_weights['weights'] = standardized_weights
    unique_labels = []
    unique_proportions = []
    for x in list(labels_weights.iloc[:, 0]):
        if x not in unique_labels:
            unique_labels.append(x)
    for y in unique_labels:
        temp = labels_weights[labels_weights['labels'] == y]
        unique_proportions.append(sum(list(temp['weights'])))
    return unique_proportions

def uncertainty_method(method, labels_weights):
    proportions = label_proportions(labels_weights)
    if method == 'entropy':
        transformed_proportions = [ -x*math.log(x, 2) for x in proportions]
        entropy = sum(transformed_proportions)
        return entropy
    elif method == 'majority_error':
        return 1 - max(proportions)
    elif method == "gini_index":
        squared_proportions = [x**2 for x in proportions]
        gini_index = 1 - sum(squared_proportions)
        return gini_index       
    else:
        sys.exit("Invalid method; choose from 'entropy', 'majority_error', or 'gini_index'.")

def get_unique_values(data):
    unique_values_dict = {}
    for x in list(data.columns):
        values = []
        for y in list(data[x]):
            if y not in values:
                values.append(y)
        unique_values_dict[x] = values
    return unique_values_dict

def get_unique_values_list(array):
    unique_values = []
    for x in array:
        if x not in unique_values:
            unique_values.append(x)
    return unique_values

def select_attribute(data, target, method, weights):
    base_entropy = uncertainty_method(method, data[[target, weights]])
    information_gain_dict = {}
    total_weight = sum(data['weights'])
    unique_values_dict = get_unique_values(data)
    non_target_features = list(data.columns)
    non_target_features.remove(target)
    non_target_features.remove(weights)
    for x in non_target_features:
        feature_unique_values = unique_values_dict[x]
        information_gain = base_entropy
        for y in feature_unique_values:
            filtered_data_by_feature = data[data[x] == y]
            filtered_total_weight = sum(filtered_data_by_feature['weights'])
            information_gain -= uncertainty_method(method, filtered_data_by_feature[[target, weights]])*(filtered_total_weight/total_weight)
        information_gain_dict[x] = information_gain
    selected_attribute = max(information_gain_dict, key = information_gain_dict.get)
    return [selected_attribute, unique_values_dict[selected_attribute]]

def add_leaf_condition(data, target, node, max_depth):
    
    if len(data.columns) == 2:
        return True
    elif node.depth == max_depth and max_depth > 0:
        return True
    elif len(get_unique_values(data)[target]) == 1:
        return True
    else:
        return False

def most_common_list(list):
    value_counts = {}
    for y in list:
        if y not in value_counts.keys():
            value_counts[y] = 1
        else:
            value_counts[y] += 1
    return max(value_counts, key = value_counts.get)

def most_common(labels_weights):
    labels_weights.columns = ['labels', 'weights']
    unique_values = get_unique_values_list(list(labels_weights['labels']))
    score_dict = {}
    for y in unique_values:
        temp = labels_weights[labels_weights['labels'] == y]
        weight = sum(temp['weights'])
        score_dict[y] = weight
    return max(score_dict, key = score_dict.get)
        
def build_tree(tree, data, target, weights, parent, method):
    if tree.root.most_common == None:
        tree.root.most_common = most_common_list(list(data[target]))
    if tree.rand_flag and tree.rand_flag < len(data.columns) - 2:
        features = list(data.columns)
        features.remove(target)
        features.remove(weights)
        rand_features = random.sample(features, k = tree.rand_flag)
        sample = data[rand_features + [target, weights]]
        next_branch_attribute = select_attribute(sample, target, method, weights)
    else:
        next_branch_attribute = select_attribute(data, target, method, weights)
    for i in next_branch_attribute[1]:
        new_node = tree.add_branch(parent, next_branch_attribute[0], i)
        filter_attribute = str(new_node.attribute)
        #filter_value = str(new_node.value)
        filter_value = new_node.value
        filtered_data = data[data[filter_attribute] == filter_value]
        filtered_data = filtered_data.loc[:, filtered_data.columns != new_node.attribute]
        labels_weights = filtered_data[[target, weights]]
        new_node.most_common = most_common(labels_weights)
        if add_leaf_condition(filtered_data, target, new_node, tree.max_depth):
            tree.add_leaf(new_node, new_node.most_common)
        else:
            build_tree(tree, filtered_data, target, weights, new_node, method)

def get_tree_predictions(tree, data):
    predictions = []
    for i in range(len(data)):
        current_node = tree.root
        unseen_trigger = False
        while current_node.child[0].type != 'leaf':
            current_node_children = current_node.child
            current_attribute = current_node.child[0].attribute
            data_row_value = data.loc[i, current_attribute]

            children_values = []
            for k in current_node_children:
                children_values.append(k.value)
            if data_row_value not in children_values:
                predictions.append(current_node.most_common)
                unseen_trigger = True
                break

            for j in current_node_children:
                if j.value == data_row_value:
                    current_node = j
                    break

        if unseen_trigger == False:
            predictions.append(j.child[0].value)
    return predictions

def get_test_accuracy(data, target, predictions):
    num_of_rows = len(predictions)
    actual = list(data.loc[:, target])
    num_correct = 0
    for i in range(num_of_rows):
        if actual[i] == predictions[i]:
            num_correct += 1
    return 100*num_correct/num_of_rows

def numeric_to_binary(data, numeric_cols):
    for column in numeric_cols:
        median = data.iloc[:, column].median()
        for i in range(len(data)):
            if data.iat[i, column] > median:
                data.iat[i, column] = 'G'
            else:
                data.iat[i, column] = 'L'
    return data

def replace_unknown(data):
    for column in data.columns:
        most_common_value = most_common(list(data.loc[:, column]))
        for i in range(len(data)):
            if data.at[i, column] == 'unknown':
                data.at[i, column] = most_common_value
    return data

def adaboost_target_encoder(target, pos, neg):
    for m in range(len(target)):
        if target[m] == pos:
            target[m] = 1
        else: 
            target[m] = -1
    return target

def build_decision_tree_adaboost_model(data, target, method, num_iterations, depth = 1):
    data['weights'] = [1/len(data)] * len(data)
    alpha_list = []
    trees = []
    for i in range(num_iterations):
        print("Building trees... " + f"{(i+1)/num_iterations*100:.1f} %", end="\r")
        sys.stdout.flush()
        temp = DecisionTree(depth)
        build_tree(temp, data, target, 'weights', temp.root, method)
        trees.append(temp)
        predictions = get_tree_predictions(temp, data)
        actual = list(data[target])
        error = 0
        incorrect_count = 0
        accuracy_df = pd.DataFrame(list(zip(data['weights'], actual, predictions)), columns = ['weights', 'actual', 'predictions'])
        for k in range(len(accuracy_df)):
            if accuracy_df.loc[k, 'actual'] != accuracy_df.loc[k, 'predictions']:
                error += accuracy_df.loc[k, 'weights']
                incorrect_count += 1
        alpha = 0.5 * math.log((1 - error)/error)
        alpha_list.append(alpha)
        new_weights = list(data['weights'])
        target_vector = list(data[target])
        for j in range(len(new_weights)):
            new_weights[j] = new_weights[j] * math.exp(-alpha * target_vector[j] * predictions[j])
        z = sum(new_weights)
        new_weights = [x / z for x in new_weights]
        data['weights'] = new_weights  
    return trees, alpha_list

def get_adaboost_accuracy(trees, alpha_list, data, target, verbose = True):
    actual = np.array(data[target])
    all_data_predictions = []
    for t in range(len(trees)):
        if verbose:
            print("Generating predictions... " + f"{(t+1)/len(trees)*100:.1f} %", end="\r")
            sys.stdout.flush()
        all_data_predictions.append(alpha_list[t] * np.array(get_tree_predictions(trees[t], data)))
        if t > 0:
            all_data_predictions[t] = all_data_predictions[t] + all_data_predictions[t-1]
    print("Getting accuracy rates...")
    sys.stdout.flush()
    prediction_array = np.sign(np.column_stack(all_data_predictions))
    accuracy_array = (np.matmul(actual, prediction_array) + len(data)) / (2 * len(data))
    iters = []
    for j in range(len(trees)):
        iters.append(j + 1)
    accuracy_dict = {'Iteration': iters, 'Accuracy': accuracy_array}
    accuracy_df = pd.DataFrame.from_dict(accuracy_dict, orient = 'columns')
    return accuracy_df

def build_bagged_decision_tree_model(data, target, method, num_iterations, bag_size, rand_flag = False, verbose = True):
    dummy_weights = [1] * len(data)
    data['weights'] = dummy_weights
    trees = []
    for i in range(num_iterations):
        if verbose:
            print("Building trees... " + f"{(i+1)/num_iterations*100:.1f} %", end="\r")
            sys.stdout.flush()
        sample_indices = random.choices(range(len(data)), k = bag_size)
        sample_list = []
        for j in sample_indices:
            new_entry = list(data.loc[j, :])
            sample_list.append(new_entry)
        sample = pd.DataFrame(sample_list, columns = data.columns)
        tree = DecisionTree(0, rand_flag = rand_flag)
        build_tree(tree, sample, target, 'weights', tree.root, method)
        trees.append(tree)
    return trees

def get_bagged_accuracy(trees, data, target, iter_accuracies = True, verbose = True):
    actual = list(data[target])
    test_predictions = []
    for t in range(len(trees)):
        if verbose:
            print("Generating predictions... " + f"{(t+1)/len(trees)*100:.1f} %", end="\r")
            sys.stdout.flush()
        test_predictions.append(get_tree_predictions(trees[t], data))
    test_accuracy = []
    test_prediction_df = pd.DataFrame(test_predictions)
    if iter_accuracies:
        for k in range(test_prediction_df.shape[0]):
            if verbose:
                print("Generating accuracy metrics... " + f"{(k+1)/test_prediction_df.shape[0]*100:.1f} %", end="\r")
                sys.stdout.flush()
            iter_test_predictions = test_prediction_df.iloc[0:k+1, :].mode()
            iter_test_predictions = list(iter_test_predictions.loc[0, :])
            test_correct = 0
            for i in range(len(actual)):
                if iter_test_predictions[i] == actual[i]:
                    test_correct += 1
            test_accuracy.append(test_correct / len(actual))
        iters = []
        for j in range(len(trees)):
            iters.append(j + 1)
        accuracy_dict = {'Iterations': iters, 'Accuracy': test_accuracy}
        accuracy_df = pd.DataFrame.from_dict(accuracy_dict, orient = 'columns')
        return accuracy_df
    test_predictions = test_prediction_df.mode()
    test_predictions = list(test_predictions.loc[0, :])
    test_correct = 0
    for i in range(len(actual)):
        if test_predictions[i] == actual[i]:
            test_correct += 1
    test_accuracy = test_correct / len(actual)
    return [test_accuracy, test_predictions]
    


