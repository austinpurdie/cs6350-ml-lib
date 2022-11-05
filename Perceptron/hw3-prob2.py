import pandas as pd
import numpy as np
import random
import sys

def perceptron(data, target, r, epochs, type = 'standard'):
    n = len(data)
    target_array = data[:, target]
    examples_array = np.delete(data, target, 1)
    w = np.array([0] * (np.shape(data)[1] - 1))
    if type == 'voted':
        w_array = np.array([])
        c_array = np.array([])
        c = 0
    for _ in range(epochs):
        order = random.sample(range(n), k = n)
        for j in order:
            temp = examples_array[j, :]
            pred = np.sign(np.dot(temp, w))
            if pred != target_array[j]:
                w = w + r * target_array[j] * temp
                if type == 'voted':
                    c = 1
                    c_array = np.append(c_array, c)
                    if len(w_array) == 0:
                        w_array = np.array([w])
                    else:
                        w_array = np.vstack((w_array, w))
            else:
                if type == 'voted':
                    c += 1
                    c_array = np.append(c_array, c)
                    if len(w_array) == 0:
                        w_array = np.array([w])
                    else:
                        w_array = np.vstack((w_array, w))
    if type == 'voted':
        return w_array, c_array
    if type =='standard':
        return w

def get_perceptron_accuracy(data, target, w, c = None, type = 'standard'):
    n = len(data)
    target_array = data[:, target]
    examples_array = np.delete(data, target, 1)
    if type == 'standard':
        prediction_array = np.sign(np.matmul(examples_array, w))
    if type != 'standard':
        prediction_array = np.array([])
        for j in range(len(examples_array)):
            temp_pred = 0
            for k in range(len(w)):
                if type == 'voted':
                    temp_pred += c[k] * np.sign(np.dot(examples_array[j], w[k]))
                if type == 'average':
                    temp_pred += np.sign(c[k] * np.dot(examples_array[j], w[k]))
            prediction_array = np.append(prediction_array, np.sign(temp_pred))
    num_correct = 0
    for i in range(n):
        if target_array[i] == prediction_array[i]:
            num_correct += 1
    accuracy = num_correct / n
    if type == 'standard':
        print("The learned weight vector is: \n" + str(w))
        sys.stdout.flush()
    if type == 'average':
        average_w = np.sum(w, axis = 0)
        print("The learned average weight vector is: \n" + str(average_w))
        sys.stdout.flush()
    print("The test accuracy is: " + str(accuracy * 100) + '%')
    sys.stdout.flush()
    if type == 'voted':
        w_df = pd.DataFrame(w, columns = ['w1', 'w2', 'w3', 'w4', 'w5'])
        w_df['vote'] = c
        w_df.to_csv('prob2b-report.csv')
    return accuracy


train_data = np.genfromtxt('Perceptron/Data/train_processed.csv', delimiter = ',')

test_data = np.genfromtxt('Perceptron/Data/test_processed.csv', delimiter = ',')

print("\nStandard Perceptron:")
sys.stdout.flush()

standard_perceptron = perceptron(train_data, 5, 0.1, 10, 'standard')

standard_accuracy = get_perceptron_accuracy(test_data, 5, standard_perceptron, type = 'standard')

print("\nVoted Perceptron")
sys.stdout.flush()

voted_perceptron = perceptron(train_data, 5, 0.1, 10, type = 'voted')

voted_accuracy = get_perceptron_accuracy(test_data, 5, voted_perceptron[0], voted_perceptron[1], 'voted')

print('\nAverage Perceptron')
sys.stdout.flush()

average_accuracy = get_perceptron_accuracy(test_data, 5, voted_perceptron[0], voted_perceptron[1], 'average')