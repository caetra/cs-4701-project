# modification of neural network code to only be able to have 2 hidden layers

import numpy as np

# Sigmoid function for activation
def sigmoid(x):
    return 1/(1 + np.exp(-x))
    
# Derivative of sigmoid for back-propagating
# Actually has a pretty cool derivative
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))
    
class NeuralNetwork_2HL:
    
    def __init__(self, num_inputs, num_hidden_first, num_hidden_second, num_outputs):
        self.num_inputs = num_inputs
        self.num_hidden_first = num_hidden_first
        self.num_hidden_second = num_hidden_second
        self.num_outputs = num_outputs
        
        # +1 for bias
        self.weights = [np.random.rand(self.num_inputs + 1, self.num_hidden_first), np.random.rand(self.num_hidden_first, self.num_hidden_second), np.random.rand(self.num_hidden_second, self.num_outputs)]
        
    def feed_forward(self, inputs):
        # add 1 to end of inputs for bias
        inputs = np.append(inputs, 1.)
        
        sums = [np.random.rand(self.num_hidden_first), np.random.rand(self.num_hidden_second), np.random.rand(self.num_outputs)]
        activations = [np.random.rand(self.num_hidden_first), np.random.rand(self.num_hidden_second), np.random.rand(self.num_outputs)]
        
        for neuron in range(self.num_hidden_first):
            sum = 0
            for input in range(self.num_inputs + 1):
                sum += (self.weights[0][input, neuron] * inputs[input])
            sums[0][neuron] = sum
            activations[0][neuron] = sigmoid(sum)
            
        for next_neuron in range(self.num_hidden_second):
            sum = 0
            for prev_neuron in range(self.num_hidden_second):
                sum += (self.weights[1][prev_neuron, next_neuron] * activations[0][prev_neuron])
            sums[1][next_neuron] = sum
            activations[1][next_neuron] = sigmoid(sum)
            
        for output in range(self.num_outputs):
            sum = 0
            for neuron in range(self.num_hidden_second):
                sum += (self.weights[2][neuron, output] * activations[1][neuron])
            sums[2][output] = sum
            activations[2][output] = sigmoid(sum)
            
        return (sums, activations)
        
    def back_propagate(self, inputs, given_outputs, learning_rate):
        (sums, activations) = self.feed_forward(inputs)
        inputs = np.append(inputs, 1.)
        
        deltas = [np.random.rand(self.num_hidden_first), np.random.rand(self.num_hidden_second), np.random.rand(self.num_outputs)]
        
        # back propagate deltas
        for output in range(self.num_outputs):
            deltas[2][output] = sigmoid_derivative(sums[2][output]) * (given_outputs[output] - activations[2][output])
            
        for neuron in range(self.num_hidden_second):
            sum = 0
            for output in range(self.num_outputs):
                sum += (self.weights[2][neuron, output] * deltas[2][output])
            deltas[1][neuron] = (sigmoid_derivative(sums[1][neuron]) * sum)
            
        for prev_neuron in range(self.num_hidden_first):
            sum = 0
            for next_neuron in range(self.num_hidden_second):
                sum += (self.weights[1][prev_neuron, next_neuron] * deltas[1][next_neuron])
            deltas[0][prev_neuron] = (sigmoid_derivative(sums[0][prev_neuron]) * sum)
            
        # use deltas to update weights
        for input in range(self.num_inputs + 1):
            for neuron in range(self.num_hidden_first):
                self.weights[0][input, neuron] += (learning_rate * inputs[input] * deltas[0][neuron])
                
        for prev_neuron in range(self.num_hidden_first):
            for next_neuron in range(self.num_hidden_second):
                self.weights[1][prev_neuron, next_neuron] += (learning_rate * activations[0][prev_neuron] * deltas[1][next_neuron])

        for neuron in range(self.num_hidden_second):
            for output in range(self.num_outputs):
                self.weights[2][neuron, output] += (learning_rate * activations[1][neuron] * deltas[2][output])
                
                
    # Train a network by repeatedly backpropagating all the examples
    def train_network(self, num_examples, inputs_list, outputs_list, num_iterations, learning_rate):
        for i in range(num_iterations):
            for j in range(num_examples):
                self.back_propagate(inputs_list[j], outputs_list[j], learning_rate)  
                
    def get_output(self, inputs):
        return self.feed_forward(inputs)[1][2]
 
# main program ########################################################################## 
# xor_net = NeuralNetwork_2HL(2, 3, 3, 1)
# a = np.array([1.,1.])
# b = np.array([1.,0.])
# c = np.array([0.,1.])
# d = np.array([0.,0.])
# examples = [a, b, c, d]
# labels = [np.array([0.]), np.array([1.]), np.array([1.]), np.array([0.])]

# xor_net.train_network(4, examples, labels, 5000, 0.3)

# print xor_net.get_output(a)
# print xor_net.get_output(b)
# print xor_net.get_output(c)
# print xor_net.get_output(d)