from cProfile import label
import math
import pandas as pd
import numpy as np


class DecisionTreeNode:
    def __init__(self, attribute, value, type, parent):
        self.attribute = attribute # which attribute the node is giving instructions for (e.g., outlook, humidity)
        self.value = value # the value of the attribute the node is giving instructions for (e.g., the outlook is //sunny//)
        self.type = type # whether the node is a root, branch, or leaf
        self.parent = parent # the node's parent node
        self.child = [] # a list of nodes that this node points to in the tree
        self.depth = 0

class DecisionTree:
    def __init__(self, max_depth):
        self.root = DecisionTreeNode(None, None, "root", None)
        self.max_depth = max_depth
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

def label_proportions(labels):
    unique_labels = []
    unique_proportions = []
    for x in labels:
        if x not in unique_labels:
            unique_labels.append(x)
    for y in unique_labels:
        unique_proportions.append(labels.count(y)/len(labels))
    return unique_proportions

### TODO : YOU MAY DELETE THE entropy(), majority_error(), and gini_index() FUNCTIONS IF THEY SEEM TO WORK AT THE END OF TESTING.
def entropy(labels):
    proportions = label_proportions(labels)
    transformed_proportions = [ -x*math.log(x, 2) for x in proportions]
    entropy = sum(transformed_proportions)
    return entropy

def majority_error(labels):
    proportions = label_proportions(labels)
    return 1 - max(proportions)

def gini_index(labels):
    proportions = label_proportions(labels)
    squared_proportions = [x**2 for x in proportions]
    gini_index = 1 - sum(squared_proportions)
    return gini_index

def uncertainty_method(method, labels):
    proportions = label_proportions(labels)
    if method == 'entropy':
        transformed_proportions = [ -x*math.log(x, 2) for x in proportions]
        entropy = sum(transformed_proportions)
        return entropy
    elif method == 'majority_error':
        proportions = label_proportions(labels)
        return 1 - max(proportions)
    elif method == "gini_index":
        squared_proportions = [x**2 for x in proportions]
        gini_index = 1 - sum(squared_proportions)
        return gini_index       
    else:
        print("Invalid method; choose from 'entropy', 'majority_error', or 'gini_index'.")

def get_unique_values(data):
    unique_values_dict = {}
    for x in list(data.columns):
        values = []
        for y in list(data[x]):
            if y not in values:
                values.append(y)
        unique_values_dict[x] = values
    return unique_values_dict
        
### TODO: DELETE THIS FUNCTION IF select_attribute() SEEMS TO WORK
def select_attribute_entropy(data, target):
    base_entropy = entropy(list(data[target]))
    information_gain_dict = {}
    total_records = data.shape[0]
    unique_values_dict = get_unique_values(data)
    non_target_features = list(data.columns)
    non_target_features.remove(target)
    for x in non_target_features:
        feature_unique_values = unique_values_dict[x]
        information_gain = base_entropy
        for y in feature_unique_values:
            filtered_data_by_feature = data[data[x] == y]
            filtered_total_records = filtered_data_by_feature.shape[0]
            information_gain -= entropy(list(filtered_data_by_feature[target]))*(filtered_total_records/total_records)
        information_gain_dict[x] = information_gain
    selected_attribute = max(information_gain_dict, key = information_gain_dict.get)
    return selected_attribute

def select_attribute(data, target, method):
    base_entropy = uncertainty_method(method, list(data[target]))
    information_gain_dict = {}
    total_records = data.shape[0]
    unique_values_dict = get_unique_values(data)
    non_target_features = list(data.columns)
    non_target_features.remove(target)
    for x in non_target_features:
        feature_unique_values = unique_values_dict[x]
        information_gain = base_entropy
        for y in feature_unique_values:
            filtered_data_by_feature = data[data[x] == y]
            filtered_total_records = filtered_data_by_feature.shape[0]
            information_gain -= uncertainty_method(method, list(filtered_data_by_feature[target]))*(filtered_total_records/total_records)
        information_gain_dict[x] = information_gain
    selected_attribute = max(information_gain_dict, key = information_gain_dict.get)
    return [selected_attribute, unique_values_dict[selected_attribute]]

# We will add a leaf when one of three conditions apply: 1) the max depth has been reached, 2) the target column values are all equal, or 3) there are no more features to select from. In cases 1 and 3, the leaf will be attached with a value equal to the most common value of the labels column.

def add_leaf_condition(data, target, node, max_depth):
    
    if len(data.columns) == 1:
        return True
    elif node.depth == max_depth:
        return True
    elif len(get_unique_values(data)[target]) == 1:
        return True
    else:
        return False

def most_common(list):
    value_counts = {}
    for y in list:
        if y not in value_counts.keys():
            value_counts[y] = 1
        else:
            value_counts[y] += 1
    return max(value_counts, key = value_counts.get)

def build_tree(tree, data, target, parent, method, max_depth):
    next_branch_attribute = select_attribute(data, target, method)
    counter = 0
    for i in next_branch_attribute[1]:
        tree.add_branch(parent, next_branch_attribute[0], i)
        counter += 1
    for node in parent.child:
        filter_attribute = str(node.attribute)
        filter_value = str(node.value)
        filtered_data = data[data[filter_attribute] == filter_value]
        filtered_data = filtered_data.loc[:, filtered_data.columns != node.attribute]
        counter -= 1
        if add_leaf_condition(filtered_data, target, node, max_depth):
            leaf_value = most_common(list(filtered_data.loc[:, target]))
            tree.add_leaf(node, leaf_value)
            if counter == 0:
                break
        else:
            parent = node
            return build_tree(tree, filtered_data, target, parent, method, max_depth)

def build_tree2(tree, data, target, parent, method, max_depth):
    next_branch_attribute = select_attribute(data, target, method)
    for i in next_branch_attribute[1]:
        new_node = tree.add_branch(parent, next_branch_attribute[0], i)
        filter_attribute = str(new_node.attribute)
        filter_value = str(new_node.value)
        filtered_data = data[data[filter_attribute] == filter_value]
        filtered_data = filtered_data.loc[:, filtered_data.columns != new_node.attribute]
        if add_leaf_condition(filtered_data, target, new_node, max_depth):
            leaf_value = most_common(list(filtered_data.loc[:, target]))
            tree.add_leaf(new_node, leaf_value)
        else:
            build_tree2(tree, filtered_data, target, new_node, method, max_depth)

def get_tree_predictions(tree, data):
    predictions = []
    for i in range(len(data)):
        current_node = tree.root
        while current_node.child[0].type != 'leaf':
            current_node_children = current_node.child
            current_attribute = current_node.child[0].attribute
            data_row_value = data.loc[i, current_attribute]
            for j in current_node_children:
                if j.value == data_row_value:
                    current_node = j
                    break
        predictions.append(j.child[0].value)
    return predictions

def get_test_accuracy(data, target, predictions):
    num_of_rows = len(predictions)
    actual = list(data.loc[:, target])
    num_correct = 0
    for i in range(len(predictions)):
        if actual[i] == predictions [i]:
            num_correct += 1
    return 100*num_correct/num_of_rows
        



df = read_data("cs6350-ml-lib/DecisionTree/tennis_test.csv", ['O', 'T', 'H', 'W', 'P'])
tree = DecisionTree(10)

build_tree2(tree, df, 'P', tree.root, 'gini_index', tree.max_depth)

print(tree.node_count)

print("The first node branching off the root of the tree is " + tree.root.child[0].attribute + " = " + tree.root.child[0].value + ". Its depth is " + str(tree.root.child[0].depth) + ".")
print("Its first child is " + tree.root.child[0].child[0].attribute + " = " + tree.root.child[0].child[0].attribute + ". Its depth is " + str(tree.root.child[0].child[0].depth) + ". This node has a leaf with value " + tree.root.child[0].child[0].child[0].value + ".")
print("The second node branching off the root is " + tree.root.child[1].attribute + " = " + tree.root.child[1].value + ". It has " + str(len(tree.root.child[1].child)) + " child of type " + tree.root.child[1].child[0].type + ". Its child's value is " + tree.root.child[1].child[0].value + ".")

print("The third node branching off the root is " + tree.root.child[2].attribute + " = " + tree.root.child[2].value + ". It has " + str(len(tree.root.child[2].child)) + " children. They are " + tree.root.child[2].child[0].attribute + " = " + tree.root.child[2].child[0].value + " and " + tree.root.child[2].child[1].attribute + " = " + tree.root.child[2].child[1].value + ". The first has " + str(len(tree.root.child[2].child[0].child)) + " child and the second also has " + str(len(tree.root.child[2].child[1].child)) + " child. Respectively, their values are " + tree.root.child[2].child[0].child[0].value + " and " + tree.root.child[2].child[1].child[0].value + ".")

print(tree.root.child[2].child[1].child[0].child)

test_predictions = get_tree_predictions(tree, df)

print(test_predictions)

test_accuracy = get_test_accuracy(df, 'P', test_predictions)

print(test_accuracy)





