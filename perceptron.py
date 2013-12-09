# Perceptron (neural net with no hidden layers)
# Can't learn things that aren't linearly separable
# But tends to work well on most real life data

import numpy as np

# Sigmoid function for activation
def sigmoid(x):
    return 1/(1 + np.exp(-x))
    
# Derivative of sigmoid for back-propagating
# Actually has a pretty cool derivative
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))
    
class Perceptron:

    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        
        self.weights = np.random.rand(self.num_inputs + 1, self.num_outputs)
        
    def feed_forward(self, inputs):
        inputs = np.append(inputs, 1.) # bias
        sums = np.random.rand(self.num_outputs)
        activations = np.random.rand(self.num_outputs)
        
        for output in range(self.num_outputs):
            sum = 0
            for input in range(self.num_inputs + 1):
                sum += self.weights[input, output] * inputs[input]
            sums[output] = sum
            activations[output] = sigmoid(sum)
            
        return (sums, activations)
        
    def back_propagate(self, inputs, given_outputs, learning_rate):
        (sums, activations) = self.feed_forward(inputs)
        inputs = np.append(inputs, 1.)
        deltas = np.random.rand(self.num_outputs)
        
        for output in range(self.num_outputs):
            deltas[output] = sigmoid_derivative(sums[output]) * (given_outputs[output] - activations[output])
            
        for input in range(self.num_inputs + 1):
            for output in range(self.num_outputs):
                self.weights[input, output] += (learning_rate * inputs[input] * deltas[output])
                
    def train_network(self, num_examples, inputs_list, outputs_list, num_iterations, learning_rate):
        for i in range(num_iterations):
            for j in range(num_examples):
                self.back_propagate(inputs_list[j], outputs_list[j], learning_rate)  
                
    def get_output(self, inputs):
        return self.feed_forward(inputs)[1]
        
#### Main program ####

print "doing and"
and_net = Perceptron(2, 1)
a = np.array([1.,1.])
b = np.array([1.,0.])
c = np.array([0.,1.])
d = np.array([0.,0.])
examples = [a, b, c, d]
labels = [np.array([1.]), np.array([0.]), np.array([0.]), np.array([0.])]

and_net.train_network(4, examples, labels, 5000, 0.3)

print and_net.get_output(a)
print and_net.get_output(b)
print and_net.get_output(c)
print and_net.get_output(d)

print "doing or"
or_net = Perceptron(2, 1)
a = np.array([1.,1.])
b = np.array([1.,0.])
c = np.array([0.,1.])
d = np.array([0.,0.])
examples = [a, b, c, d]
labels = [np.array([1.]), np.array([1.]), np.array([1.]), np.array([0.])]

or_net.train_network(4, examples, labels, 5000, 0.3)

print or_net.get_output(a)
print or_net.get_output(b)
print or_net.get_output(c)
print or_net.get_output(d)