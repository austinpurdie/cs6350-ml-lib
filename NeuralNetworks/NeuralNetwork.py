# I wrote this code with guidance from this article:
# http://neuralnetworksanddeeplearning.com/chap2.html
# I use some strategies that the author uses in his code, but the code below
# is mine.

import numpy as np
import pandas as pd
import random
import sys

def sigma(x):
    return 1/(1 + np.exp(-x))

def sigma_d(x):
    return sigma(x) * (1 - sigma(x))

class NeuralNetwork:
    def __init__(self, architecture, initialize):
        # architecture is a list of layer sizes; the first item in the list
        # should be the input size, all of the items between the first and 
        # last should be the width of each hidden layer, and the last item
        # should be 1 to represent the output layer

        self.architecture = architecture
        self.depth = len(architecture)

        # when initialize is 'zero', the weights and bias are all 
        # initialized as zeros; when it is 'gaussian', they are initialized
        # with standard normal random variables

        if initialize == 'gaussian':

            self.weight = [np.random.randn(a, b) for a, b in zip(self.architecture[0:self.depth-1], self.architecture[-(self.depth - 1):])]
            self.bias = [np.zeros((1, self.architecture[1]))]
            for layer in self.architecture[2:self.depth]:
                self.bias.append(np.random.randn(1, layer))

        elif initialize == 'zero':
            self.weight = [np.zeros((a, b)) for a, b in zip(self.architecture[0:self.depth-1], self.architecture[-(self.depth - 1):])]
            self.bias = [np.zeros((1, c)) for c in self.architecture[1:]]

        else:
            sys.exit("Invalid initialize argument; use 'gaussian' or 'zero'")

    def back_propagation(self, example, label):
        activations = [example]
        z_values = [example]
        for layer in range(self.depth-1):
            z = np.dot(activations[-1], self.weight[layer]) + self.bias[layer]
            z_values.append(z)
            activation = sigma(z)
            activations.append(activation)

        top_delta = activations[-1] - label
        delta_values = [top_delta]
        for layer in range(1, self.depth-1):
            delta = (np.dot(delta_values[-1], np.transpose(self.weight[-layer]))) * sigma_d(z_values[-(layer + 1)])
            delta_values.append(delta)
        delta_values = delta_values[::-1]

        weight_partials = [np.transpose(np.outer(a, b)) for a, b in zip(delta_values, activations[0:self.depth])]
        bias_partials = delta_values

        return weight_partials, bias_partials

    def stoch_grad(self, data, labels, epochs, gamma_0, d):
        n_rows = len(data)
        gamma = gamma_0
        t = 0
        for _ in range(epochs):
            sample_indices = random.sample(range(n_rows), n_rows)
            for i in sample_indices:
                x = data[i]
                y = labels[i]
                grad = self.back_propagation(x, y)
                self.weight = [a - gamma * b for a, b in zip(self.weight, grad[0])]
                self.bias = [a - gamma * b for a, b in zip(self.bias, grad[1])]
                t += 1
                gamma = gamma_0/(1 + (gamma_0/d) * t)
                
    def get_output(self, example):
        result = example
        for layer in range(self.depth-1):
            result = np.dot(result, self.weight[layer]) + self.bias[layer]
        return result


    def get_accuracy(self, examples, labels):
        num_correct = 0
        out_of_bounds_greater = 0
        out_of_bounds_less = 0

        for i in range(len(examples)):
            output = self.get_output(examples[i])

            if output > 1:
                out_of_bounds_greater += 1
                output = 1
            elif output <= 1 and output >= 0.5:
                output = 1
            elif output < 0:
                out_of_bounds_less += 1
                output = 0
            elif output >= 0 and output < 0.5:
                output = 0
            
            if output == labels[i]:
                num_correct += 1

        accuracy = num_correct / len(examples)

        return accuracy

def nn_3layer(widths, input_size, epochs, gamma_0, d, train_examples, train_labels, test_examples, test_labels, initialize):
    networks = []
    train_accuracy_list = []
    test_accuracy_list = []
    for i in widths:
        print("Now training width = " + str(i))
        sys.stdout.flush()
        networks.append(NeuralNetwork([input_size, i, i, 1], initialize))
        networks[-1].stoch_grad(train_examples, train_labels, epochs, gamma_0, d)
        train_accuracy_list.append(networks[-1].get_accuracy(train_examples, train_labels))
        test_accuracy_list.append(networks[-1].get_accuracy(test_examples, test_labels))
    
    results = pd.DataFrame([widths, train_accuracy_list, test_accuracy_list])
    results = results.transpose()
    results.columns = ['Width', 'Training Accuracy', 'Test Accuracy']

    return results