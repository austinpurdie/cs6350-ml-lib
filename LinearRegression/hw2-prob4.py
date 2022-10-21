import numpy as np
import pandas as pd
import sys
import random

def f(w, x, y):
    result_list = []
    for i in range(len(x)):
        iter = (y[i] - np.matmul(np.transpose(w), x[i])) ** 2
        result_list.append(iter)
    result = 0.5 * sum(result_list)
    return result

def gradf(w, x, y):
    result = np.array([])
    for j in range(len(w)):
        result_list = []
        for i in range(len(x)):
            iter = (y[i] - np.matmul(np.transpose(w), x[i])) * x[i][j]
            result_list.append(iter)
        grad_comp = -1 * sum(result_list)
        result = np.append(result, grad_comp)
    return result

def backtracking_gradient(w0, x, y, epsilon, max_iter, s, alpha, beta):
    w = w0
    gradient = gradf(w, x, y)
    value = f(w, x, y)
    iter_count = 0
    t_list = []
    value_list = []
    while np.linalg.norm(gradf(w, x, y)) > epsilon and iter_count < max_iter:
        iter_count += 1
        t = s
        while value - f(w - t * gradf(w, x, y), x, y) < -alpha * t * np.linalg.norm(gradf(w, x, y)) ** 2:
            t = beta * t
        w = w - t * gradient
        t_list.append(t)
        value_list.append(value)
        value = f(w, x, y)
        gradient = gradf(w, x, y)
        print("Iteration: " + f"{iter_count:.0f}" + "       Value: " + f"{f(w, x, y):.9f}" + "       ||grad(f)|| = " + f"{np.linalg.norm(gradient):.9f}", end="\r")
        sys.stdout.flush()
    if iter_count >= max_iter:
        sys.exit("\n \nThe algorithm did not converge before the maximum number of iterations. Last function value: " + str(value))
    else:
        print("\n \nAlgorithm converged. \nTotal Iterations: " + str(iter_count) + "\nFinal function value: " + str(value) + "\nLearned Weight Vector: " + str(w) + "\nLearning Rates: see PDF writeup for details.")
        sys.stdout.flush()
    return [w, value, iter_count, t_list, value_list]

def stochastic_backtracking_gradient(w0, x, y, r, epsilon, max_iter):
    w = w0
    iter_count = 0
    value_list = []
    # while np.linalg.norm(gradf(w, x, y)) > epsilon and iter_count < max_iter:
    while np.linalg.norm(r * gradf(w, x, y)) > epsilon and iter_count < max_iter:
        samp_index = random.sample(range(len(x)), k = len(x))
        for i in samp_index:
            iter_count += 1
            sample = np.array([x[i, :]])
            sample_output = np.array([y[i]])
            gradient = gradf(w, sample, sample_output)
            value = f(w, sample, sample_output)
            w = w - r * (gradient)
            value_list.append(value)
            value = f(w, sample, sample_output)
            print("Iteration: " + f"{iter_count:.0f}" + "       Value: " + f"{f(w, x, y):.9f}" + "       r * ||grad(f)|| = " + f"{np.linalg.norm(r * gradf(w, x, y)):.9f}", end="\r")
            sys.stdout.flush()

    if iter_count >= max_iter:
        sys.exit("\n \nThe algorithm did not converge before the maximum number of iterations. Last function value: " + str(f(w, x, y)))
    else:
        print("\n \nAlgorithm converged. \nTotal Iterations: " + str(iter_count) + "\nFinal function value: " + str(value) + "\nLearned Weight Vector: " + str(w) + "\nLearning Rate: 0.00008")
        sys.stdout.flush()
    return [w, value, iter_count, value_list]

epsilon = 1/1000000
max_iter = 1000000

colnames = ['cement', 'slag', 'fly ash', 'water', 'sp', 'coarse_aggr', 'fine_aggr', 'output']

train_data = np.genfromtxt("LinearRegression/Data/concrete-train.csv", delimiter = ",")
test_data = np.genfromtxt("LinearRegression/Data/concrete-test.csv", delimiter = ",")

train_y = train_data[:, 7]
test_y = test_data[:, 7]
train_x = train_data[:, 0:7]
test_x = test_data[:, 0:7]

w0 = np.array([0, 0, 0, 0, 0, 0, 0])

xtx_inv = np.linalg.inv(np.matmul(np.transpose(train_x), train_x))

xty = np.matmul(np.transpose(train_x), train_y)

optimum = np.matmul(xtx_inv, xty)

print("\nOptimum Analytic Solution: \n" + str(optimum))
sys.stdout.flush()

print("\nRunning batch gradient descent...")
sys.stdout.flush()
gradient_model = backtracking_gradient(w0, train_x, train_y, epsilon, max_iter, 0.1, 0.3, 0.3)
weight_vector = gradient_model[0]
test_cost = f(weight_vector, test_x, test_y)
print("Test Data Cost: " + str(test_cost))
sys.stdout.flush()
np.savetxt('grad_cost.csv', gradient_model[4], delimiter = ",")

w0 = np.array([0.9, 0.8, 0.9, 1.3, 0.1, 1.6, 1])

print("\n \nRunning stochastic gradient descent...")
sys.stdout.flush()
stochastic_gradient_model = stochastic_backtracking_gradient(w0, train_x, train_y, 0.00008, epsilon, max_iter)
weight_vector = stochastic_gradient_model[0]
test_cost = f(weight_vector, test_x, test_y)
print("Test Data Cost: " + str(test_cost))
sys.stdout.flush()
np.savetxt('stoch_grad_cost.csv', stochastic_gradient_model[3], delimiter = ",")


