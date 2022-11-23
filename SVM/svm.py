import numpy as np
import random

random.seed(15012358)

def gamma_a(gamma_0, a, t):
    gamma = gamma_0/(1 + (gamma_0/a)*t)
    return gamma

def gamma_b(gamma_0, t):
    gamma = gamma_0/(1 + t)
    return gamma

def build_svm(data, target, num_epochs, c, gamma_0, a = False):
    if a:
        gamma = gamma_a(gamma_0, a, 0)
    else:
        gamma = gamma_b(gamma_0, 0)

    w = np.array([0] * (np.shape(data)[1]))
    w0 = np.array([0] * (np.shape(data)[1]))
    n = len(data)
    obj_list = []
    for _ in range(num_epochs):
        samp_index = random.sample(range(n), k = n)
        for i in samp_index:
            if target[i] * np.dot(w, data[i]) <= 1:
                w = w - gamma * w0 + gamma * c * n * target[i] * data[i]
                w0 = np.append(w[0:np.shape(data)[1] - 1], 0)
            else: 
                w0 = (1 - gamma) * w0
            obj = 0.5 * np.dot(w0, w0) + c * n * max(0, 1 - target[i] * np.dot(w, data[i]))
            obj_list.append(obj)
    
    return w

def svm_accuracy(w, data, target):
    predictions = []
    n = len(data)
    for i in range(len(data)):
        this_prediction = np.sign(np.dot(w, data[i]))
        if this_prediction == target[i]:
            predictions.append(1)
        else:
            predictions.append(0)
    predictions = np.array(predictions)
    accuracy = np.sum(predictions)/n
    return accuracy

