#from perceptron import Perceptron
#from neural_network_1_hidden import NeuralNetwork_1HL
#from neural_network_2_hidden import NeuralNetwork_2HL

# CHANGE THESE PARAMETERS, OBSERVE RESULTS
num_iterations = 5
learning_rate = 0.3
num_neurons = 40

print('number of iterations = '+str(num_iterations))
print('learning rate = '+str(learning_rate))
print('hidden neurons = '+str(num_neurons))
import pickle

import numpy as np

import neural_network_1_hidden

# Sigmoid function for activation
def sigmoid(x):
    return 1/(1 + np.exp(-x))
    
# Derivative of sigmoid for back-propagating
# Actually has a pretty cool derivative
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))
    
f1 = open('train_data_reduced.pickle', 'r')
f2 = open('train_label.pickle', 'r')
f3 = open('test_data_reduced.pickle', 'r')
f4 = open('test_label.pickle', 'r')
train_attr = pickle.load(f1)
train_label = pickle.load(f2)
test_attr = pickle.load(f3)
test_label = pickle.load(f4)
f1.close()
f2.close()
f3.close()
f4.close()

#num_attrs = 57 # not reduced
num_attrs = 54 # reduced
train_data_size = 3000
test_data_size = 1601

# all entries in the lists have to be in numpy format
for i in range(train_data_size):
    train_attr[i] = np.array(train_attr[i])
    train_label[i] = np.array([train_label[i]])
for i in range(test_data_size):
    test_attr[i] = np.array(test_attr[i])
    test_label[i] = np.array([test_label[i]])

print('Loaded data')
print('Training neural net, 1 layer')

per = neural_network_1_hidden.NeuralNetwork_1HL(num_attrs, num_neurons, 1)
per.train_network(train_data_size, train_attr, train_label, num_iterations, learning_rate)

print('Trained net')

# check training accuracy
num_train_correct = 0
for i in range(train_data_size):
    if per.get_output(train_attr[i])[0] > 0.5:
        if train_label[i][0] == 1:
            num_train_correct += 1
    else:
        if train_label[i][0] == 0:
            num_train_correct += 1

train_accuracy = num_train_correct/float(train_data_size)
print(str(train_accuracy)+' training accuracy for perceptron')
print(str(num_train_correct)+ ' of ' + str(train_data_size))

# check test accuracy
num_test_correct = 0
for i in range(test_data_size):
    if per.get_output(test_attr[i])[0] > 0.5:
        if test_label[i][0] == 1:
            num_test_correct += 1
    else:
        if test_label[i][0] == 0:
            num_test_correct += 1

test_accuracy = num_test_correct/float(test_data_size)
print(str(test_accuracy)+' test accuracy for perceptron')
print(str(num_test_correct)+ ' of ' + str(test_data_size))

