import DecisionTree as dt
import pandas as pd

tree1_e = dt.DecisionTree(1)
tree2_e = dt.DecisionTree(2)
tree3_e = dt.DecisionTree(3)
tree4_e = dt.DecisionTree(4)
tree5_e = dt.DecisionTree(5)
tree6_e = dt.DecisionTree(6)

trees_e = [tree1_e, tree2_e, tree3_e, tree4_e, tree5_e, tree6_e]

tree1_m = dt.DecisionTree(1)
tree2_m = dt.DecisionTree(2)
tree3_m = dt.DecisionTree(3)
tree4_m = dt.DecisionTree(4)
tree5_m = dt.DecisionTree(5)
tree6_m = dt.DecisionTree(6)

trees_m = [tree1_m, tree2_m, tree3_m, tree4_m, tree5_m, tree6_m]

tree1_g = dt.DecisionTree(1)
tree2_g = dt.DecisionTree(2)
tree3_g = dt.DecisionTree(3)
tree4_g = dt.DecisionTree(4)
tree5_g = dt.DecisionTree(5)
tree6_g = dt.DecisionTree(6)

trees_g = [tree1_g, tree2_g, tree3_g, tree4_g, tree5_g, tree6_g]

colnames = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']

train_data = pd.read_csv("/car_train.csv", names = colnames, header = None)

test_data = pd.read_csv("/car_test.csv", names = colnames, header = None)

for t in trees_e:
    dt.build_tree(t, train_data, 'label', t.root, 'entropy', t.max_depth)

for t in trees_m:
    dt.build_tree(t, train_data, 'label', t.root, 'majority_error', t.max_depth)

for t in trees_g:
    dt.build_tree(t, train_data, 'label', t.root, 'gini_index', t.max_depth)

test_accuracy = pd.DataFrame(columns = ['Method', 'Max Depth', 'Accuracy Rate', 'Error Rate'])

for r in range(len(trees_e)):
    predictions = dt.get_tree_predictions(trees_e[r], test_data)
    tree_accuracy = dt.get_test_accuracy(test_data, 'label', predictions)
    test_accuracy.loc[len(test_accuracy.index)] = ['Entropy', r + 1, tree_accuracy, 100 - tree_accuracy]

for r in range(len(trees_m)):
    predictions = dt.get_tree_predictions(trees_m[r], test_data)
    tree_accuracy = dt.get_test_accuracy(test_data, 'label', dt.get_tree_predictions(trees_m[r], test_data))
    test_accuracy.loc[len(test_accuracy.index)] = ['Majority Error', r + 1, tree_accuracy, 100 - tree_accuracy]

for r in range(len(trees_g)):
    predictions = dt.get_tree_predictions(trees_g[r], test_data)
    tree_accuracy = dt.get_test_accuracy(test_data, 'label', dt.get_tree_predictions(trees_g[r], test_data))
    test_accuracy.loc[len(test_accuracy.index)] = ['Gini Index', r + 1, tree_accuracy, 100 - tree_accuracy]

print(test_accuracy)

train_accuracy = pd.DataFrame(columns = ['Method', 'Max Depth', 'Accuracy Rate', 'Error Rate'])

for r in range(len(trees_e)):
    predictions = dt.get_tree_predictions(trees_e[r], train_data)
    tree_accuracy = dt.get_test_accuracy(train_data, 'label', predictions)
    train_accuracy.loc[len(train_accuracy.index)] = ['Entropy', r + 1, tree_accuracy, 100 - tree_accuracy]

for r in range(len(trees_m)):
    predictions = dt.get_tree_predictions(trees_m[r], train_data)
    tree_accuracy = dt.get_test_accuracy(train_data, 'label', predictions)
    train_accuracy.loc[len(train_accuracy.index)] = ['Majority Error', r + 1, tree_accuracy, 100 - tree_accuracy]

for r in range(len(trees_g)):
    predictions = dt.get_tree_predictions(trees_g[r], train_data)
    tree_accuracy = dt.get_test_accuracy(train_data, 'label', predictions)
    train_accuracy.loc[len(train_accuracy.index)] = ['Gini Index', r + 1, tree_accuracy, 100 - tree_accuracy]

print(train_accuracy)





