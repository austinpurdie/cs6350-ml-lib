from NeuralNetwork import *

train_set = np.genfromtxt(fname = "NeuralNetworks/Data/train.csv", delimiter = ",")
train_examples = train_set[:, 0:4]
train_labels = train_set[:, 4]

test_set = np.genfromtxt(fname = "NeuralNetworks/Data/test.csv", delimiter = ",")
test_examples = test_set[:, 0:4]
test_labels = test_set[:, 4]

np.random.seed(571832)
random.seed(183275)

print("Training Gaussian Initialized Networks:\n")

gaussian_results = nn_3layer(widths = [5, 10, 25, 50, 100], input_size = 4, 
                    epochs = 200, gamma_0 = 1, d = 50, 
                    train_examples = train_examples, train_labels = train_labels, 
                    test_examples = test_examples, test_labels = test_labels, 
                    initialize = 'gaussian')

print("\nGaussian Initialization Results (Problem 2b)\n")
sys.stdout.flush()
print(gaussian_results)
sys.stdout.flush()

print("\nTraining Zero Initialized Networks:\n")
sys.stdout.flush()

zero_results = nn_3layer(widths = [5, 10, 25, 50, 100], input_size = 4, 
                    epochs = 200, gamma_0 = 1, d = 50, 
                    train_examples = train_examples, train_labels = train_labels, 
                    test_examples = test_examples, test_labels = test_labels, 
                    initialize = 'zero')

print("\nZero Initialization Results (Problem 2c)\n")
sys.stdout.flush()
print(zero_results)
sys.stdout.flush()