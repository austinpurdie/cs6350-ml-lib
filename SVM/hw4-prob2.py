from svm import *
import numpy as np
import random
import sys


train_data_all = np.loadtxt("SVM/Data/train.csv", delimiter = ",")
test_data_all = np.loadtxt("SVM/Data/test.csv", delimiter = ",")

train_data = train_data_all[:, 0:5]
test_data = test_data_all[:, 0:5]

train_target = train_data_all[:, 5]
test_target = test_data_all[:, 5]

w1 = build_svm(train_data, train_target, 100, c = 100/873, gamma_0 = 1, a = 0.1)

w2 = build_svm(train_data, train_target, 100, c = 500/873, gamma_0 = 1, a = 0.1)

w3 = build_svm(train_data, train_target, 100, c = 700/873, gamma_0 = 1, a = 0.1)

w4 = build_svm(train_data, train_target, 100, c = 100/873, gamma_0 = 0.1)

w5 = build_svm(train_data, train_target, 100, c = 500/873, gamma_0 = 0.1)

w6 = build_svm(train_data, train_target, 100, c = 700/873, gamma_0 = 0.1)


print("\nProblem 2(a) Results")
sys.stdout.flush()
print("\nC = 100/873")
sys.stdout.flush()
print("Learned w: " + str(w1) + "\n Test Accuracy: " + str(svm_accuracy(w1, test_data, test_target)) + "\n Train Accuracy: " + str(svm_accuracy(w1, train_data, train_target)))
sys.stdout.flush()
print("\nC = 500/873")
sys.stdout.flush()
print("Learned w: " + str(w2) + "\n Test Accuracy: " + str(svm_accuracy(w2, test_data, test_target)) + "\n Train Accuracy: " + str(svm_accuracy(w2, train_data, train_target)))
sys.stdout.flush()
print("\nC = 700/873")
sys.stdout.flush()
print("Learned w: " + str(w3) + "\n Test Accuracy: " + str(svm_accuracy(w3, test_data, test_target)) + "\n Train Accuracy: " + str(svm_accuracy(w3, train_data, train_target)))
sys.stdout.flush()

print("\nProblem 2(b) Results")
sys.stdout.flush()

print("\nC = 100/873")
sys.stdout.flush()
print("Learned w: " + str(w4) + "\n Test Accuracy: " + str(svm_accuracy(w4, test_data, test_target)) + "\n Train Accuracy: " + str(svm_accuracy(w4, train_data, train_target)))
sys.stdout.flush()
print("\nC = 500/873")
sys.stdout.flush()
print("Learned w: " + str(w5) + "\n Test Accuracy: " + str(svm_accuracy(w5, test_data, test_target)) + "\n Train Accuracy: " + str(svm_accuracy(w5, train_data, train_target)))
sys.stdout.flush()
print("\nC = 700/873")
sys.stdout.flush()
print("Learned w: " + str(w6) + "\n Test Accuracy: " + str(svm_accuracy(w6, test_data, test_target)) + "\n Train Accuracy: " + str(svm_accuracy(w6, train_data, train_target)))
sys.stdout.flush()

print("\n Problem 2(c) Results")
sys.stdout.flush()
print("The difference between the learned parameters when C = 100/873 is \n" + str(w1 - w4))
print("The difference between the learned parameters when C = 500/873 is \n" + str(w2 - w5))
print("The difference between the learned parameters when C = 700/873 is \n" + str(w3 - w6))

print("\nThe difference between the test errors when C = 100/873 is " + str(svm_accuracy(w4, test_data, test_target) - svm_accuracy(w1, test_data, test_target)))
print("\nThe difference between the train errors when C = 100/873 is " + str(svm_accuracy(w4, train_data, train_target) - svm_accuracy(w1, train_data, train_target)))

print("\nThe difference between the test errors when C = 500/873 is " + str(svm_accuracy(w5, test_data, test_target) - svm_accuracy(w2, test_data, test_target)))
print("\nThe difference between the train errors when C = 500/873 is " + str(svm_accuracy(w5, train_data, train_target) - svm_accuracy(w2, train_data, train_target)))

print("\nThe difference between the test errors when C = 700/873 is " + str(svm_accuracy(w6, test_data, test_target) - svm_accuracy(w3, test_data, test_target)))
print("\nThe difference between the train errors when C = 700/873 is " + str(svm_accuracy(w6, train_data, train_target) - svm_accuracy(w3, train_data, train_target)))

print("\nExcept in the instance of the test accuracy when C = 500/873, the schedule used in part (b) produced results with higher accuracy.")