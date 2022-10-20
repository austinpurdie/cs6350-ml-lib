import DecisionTree as dt
import pandas as pd

tree1_e = dt.DecisionTree(1)
tree2_e = dt.DecisionTree(2)
tree3_e = dt.DecisionTree(3)
tree4_e = dt.DecisionTree(4)
tree5_e = dt.DecisionTree(5)
tree6_e = dt.DecisionTree(6)
tree7_e = dt.DecisionTree(7)
tree8_e = dt.DecisionTree(8)
tree9_e = dt.DecisionTree(9)
tree10_e = dt.DecisionTree(10)
tree11_e = dt.DecisionTree(11)
tree12_e = dt.DecisionTree(12)
tree13_e = dt.DecisionTree(13)
tree14_e = dt.DecisionTree(14)
tree15_e = dt.DecisionTree(15)
tree16_e = dt.DecisionTree(16)

trees_e = [tree1_e, tree2_e, tree3_e, tree4_e, tree5_e, tree6_e, tree7_e, tree8_e, tree9_e, tree10_e, tree11_e, tree12_e, tree13_e, tree14_e, tree15_e, tree16_e]

tree1_m = dt.DecisionTree(1)
tree2_m = dt.DecisionTree(2)
tree3_m = dt.DecisionTree(3)
tree4_m = dt.DecisionTree(4)
tree5_m = dt.DecisionTree(5)
tree6_m = dt.DecisionTree(6)
tree7_m = dt.DecisionTree(7)
tree8_m = dt.DecisionTree(8)
tree9_m = dt.DecisionTree(9)
tree10_m = dt.DecisionTree(10)
tree11_m = dt.DecisionTree(11)
tree12_m = dt.DecisionTree(12)
tree13_m = dt.DecisionTree(13)
tree14_m = dt.DecisionTree(14)
tree15_m = dt.DecisionTree(15)
tree16_m = dt.DecisionTree(16)

trees_m = [tree1_m, tree2_m, tree3_m, tree4_m, tree5_m, tree6_m, tree7_m, tree8_m, tree9_m, tree10_m, tree11_m, tree12_m, tree13_m, tree14_m, tree15_m, tree16_m]

tree1_g = dt.DecisionTree(1)
tree2_g = dt.DecisionTree(2)
tree3_g = dt.DecisionTree(3)
tree4_g = dt.DecisionTree(4)
tree5_g = dt.DecisionTree(5)
tree6_g = dt.DecisionTree(6)
tree7_g = dt.DecisionTree(7)
tree8_g = dt.DecisionTree(8)
tree9_g = dt.DecisionTree(9)
tree10_g = dt.DecisionTree(10)
tree11_g = dt.DecisionTree(11)
tree12_g = dt.DecisionTree(12)
tree13_g = dt.DecisionTree(13)
tree14_g = dt.DecisionTree(14)
tree15_g = dt.DecisionTree(15)
tree16_g = dt.DecisionTree(16)


trees_g = [tree1_g, tree2_g, tree3_g, tree4_g, tree5_g, tree6_g, tree7_g, tree8_g, tree9_g, tree10_g, tree11_g, tree12_g, tree13_g, tree14_g, tree15_g, tree16_g]

colnames = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']

numeric_cols = [0, 5, 9, 11, 12, 13, 14]

train_data = dt.numeric_to_binary(pd.read_csv("DecisionTree/Data/bank-train.csv", names = colnames, header = None), numeric_cols)

test_data = dt.numeric_to_binary(pd.read_csv("DecisionTree/Data/bank-test.csv", names = colnames, header = None), numeric_cols)

for t in trees_e:
    dt.build_tree(t, train_data, 'y', t.root, 'entropy')

for t in trees_m:
    dt.build_tree(t, train_data, 'y', t.root, 'majority_error')

for t in trees_g:
    dt.build_tree(t, train_data, 'y', t.root, 'gini_index')

test_accuracy = pd.DataFrame(columns = ['Method', 'Max Depth', 'Accuracy Rate', 'Error Rate'])

for r in range(len(trees_e)):
    predictions = dt.get_tree_predictions(trees_e[r], test_data)
    tree_accuracy = dt.get_test_accuracy(test_data, 'y', predictions)
    test_accuracy.loc[len(test_accuracy.index)] = ['Entropy', r + 1, tree_accuracy, 100 - tree_accuracy]

for r in range(len(trees_m)):
    predictions = dt.get_tree_predictions(trees_m[r], test_data)
    tree_accuracy = dt.get_test_accuracy(test_data, 'y', dt.get_tree_predictions(trees_m[r], test_data))
    test_accuracy.loc[len(test_accuracy.index)] = ['Majority Error', r + 1, tree_accuracy, 100 - tree_accuracy]

for r in range(len(trees_g)):
    predictions = dt.get_tree_predictions(trees_g[r], test_data)
    tree_accuracy = dt.get_test_accuracy(test_data, 'y', dt.get_tree_predictions(trees_g[r], test_data))
    test_accuracy.loc[len(test_accuracy.index)] = ['Gini Index', r + 1, tree_accuracy, 100 - tree_accuracy]

print("Test Accuracy Table \n")
print(test_accuracy)


train_accuracy = pd.DataFrame(columns = ['Method', 'Max Depth', 'Accuracy Rate', 'Error Rate'])

for r in range(len(trees_e)):
    predictions = dt.get_tree_predictions(trees_e[r], train_data)
    tree_accuracy = dt.get_test_accuracy(train_data, 'y', predictions)
    train_accuracy.loc[len(train_accuracy.index)] = ['Entropy', r + 1, tree_accuracy, 100 - tree_accuracy]

for r in range(len(trees_m)):
    predictions = dt.get_tree_predictions(trees_m[r], train_data)
    tree_accuracy = dt.get_test_accuracy(train_data, 'y', predictions)
    train_accuracy.loc[len(train_accuracy.index)] = ['Majority Error', r + 1, tree_accuracy, 100 - tree_accuracy]

for r in range(len(trees_g)):
    predictions = dt.get_tree_predictions(trees_g[r], train_data)
    tree_accuracy = dt.get_test_accuracy(train_data, 'y', predictions)
    train_accuracy.loc[len(train_accuracy.index)] = ['Gini Index', r + 1, tree_accuracy, 100 - tree_accuracy]

print("Test Accuracy Table \n")
print(train_accuracy)
