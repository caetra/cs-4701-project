# Neural network implementation for AI Prac project
# Chris Anderson (cma227) and Alex Luo (hl532)

import numpy as np

# Sigmoid function for activation
def sigmoid(x):
    return 1/(1 + np.exp(-x))
    
# Derivative of sigmoid for back-propagating
# Actually has a pretty cool derivative
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))
 
class NeuralNetwork:
    # Multilayer feedforward neural network (multilayer perceptron)
    # Since there is no state other than the weights, everything can be represented as arrays of weights
    # For a layer, layer[a,b] = weight from neuron a in first layer to b in next layer
    
    def __init__(self, num_inputs, num_hidden_layers, neurons_per_layer, num_outputs):
        self.num_inputs = num_inputs
        self.num_hidden_layers = num_hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.num_outputs = num_outputs
        
        # Create the neural net, randomly wired
        self.weights = [] # weights between each hidden layer
        # Add input weights, num_inputs+1 (+1 for bias term) by neurons_per_layer
        self.weights.append(np.random.rand(self.num_inputs + 1, self.neurons_per_layer))
        # weights[0] is the weights from the 0th layer (inputs) to 1st hidden layer
        # weights[layer][a,b] is the weight from neuron a in layer to neuron b in the next layer
        for i in range(self.num_hidden_layers - 1):
            self.weights.append(np.random.rand(self.neurons_per_layer, self.neurons_per_layer)) # weights between hidden layers
        # output weights
        self.weights.append(np.random.rand(self.neurons_per_layer, self.num_outputs))
        # length of weights is num_hidden_layers + 1
        
    # Feed forward for computing outputs from inputs, inputs is a 1D numpy array of size num_inputs
    def feed_forward(self, inputs):
        inputs = np.append(inputs, 1.) # add the bias term
        
        # set up arrays
        sums = []
        for i in range(self.num_hidden_layers):
            sums.append(np.zeros(self.neurons_per_layer))
        sums.append(np.zeros(self.num_outputs))
        activations = []
        for i in range(self.num_hidden_layers):
            activations.append(np.zeros(self.neurons_per_layer))
        activations.append(np.zeros(self.num_outputs))
        # activations[a][b] is the value of activation function for layer a, bth neuron, same for sums
        
        # do inputs -> first hidden layer
        for next_neuron in range(self.neurons_per_layer):
            sum = 0
            for input in range(self.num_inputs + 1):
                sum += self.weights[0][input, next_neuron] * inputs[input]
            sums[0][next_neuron] = sum
            activations[0][next_neuron] = sigmoid(sum)
        
        # do hidden layers
        for layer in range(1, self.num_hidden_layers): # length of weights is num_hidden_layers + 1
            for next_neuron in range(self.neurons_per_layer): # this should work?
                sum = 0
                for prev_neuron in range(self.neurons_per_layer):
                    sum += self.weights[layer][prev_neuron, next_neuron] * activations[layer - 1][prev_neuron]
                sums[layer][next_neuron] = sum
                activations[layer][next_neuron] = sigmoid(sum)
                
        # do last hidden layer -> outputs
        for output in range(self.num_outputs):
            sum = 0
            for prev_neuron in range(self.neurons_per_layer):
                sum += self.weights[self.num_hidden_layers][prev_neuron, output] * activations[self.num_hidden_layers - 1][prev_neuron]
            sums[self.num_hidden_layers][output] = sum
            activations[self.num_hidden_layers][output] = sigmoid(sum)
                       
        return (sums, activations) # activations[self.num_hidden_layers] is the outputs, but we need the rest of this for backpropagate
        
        
    # Back propagates a single example to update weights
    # correct_outputs is a numpy array of size num_outputs
    def back_propagate(self, inputs, correct_outputs):
        # get the outputs we currently get for that input
        (sums, activations) = self.feed_forward(inputs)
        network_outputs = activations[self.num_hidden_layers]
        
        # create list to store deltas
        deltas = []
        deltas.append(np.zeros(self.num_inputs + 1))
        for i in range(self.num_hidden_layers - 1):
            deltas.append(np.zeros(self.neurons_per_layer))
        deltas.append(np.zeros(self.num_outputs))
        # so deltas[a][b] = delta of neuron b in layer a
        
        # Get deltas for the outputs
        for output in range(self.num_outputs):
            deltas[self.num_hidden_layers][output] = sigmoid_derivative(sums[self.num_hidden_layers][output]) * (correct_outputs[output] - network_outputs[output])
        
        # Propagate the deltas backward through hidden layers
        for layer in list(reversed(range(1, self.num_hidden_layers))): # do layers in reverse order
            for neuron in range(self.neurons_per_layer):
                sum = 0
                num_in_next = 0
                if layer == self.num_hidden_layers - 1:
                    num_in_next = self.num_outputs
                else:
                    num_in_next = self.neurons_per_layer
                for next_neuron in range(num_in_next):
                    sum += self.weights[layer][neuron, next_neuron] * deltas[layer + 1][next_neuron]
                deltas[layer][neuron] = sigmoid_derivative(sums[layer][neuron]) * sum
                
        # Propagate deltas for input layer
        for input in range(self.num_inputs + 1):
            sum = 0
            num_in_next = 0
            if self.num_hidden_layers == 1:
                num_in_next = self.num_outputs
            else:
                num_in_next = self.neurons_per_layer
            for neuron in range(num_in_next):
                sum += self.weights[0][input, neuron] * deltas[1][neuron]
            deltas[0][input] = sigmoid_derivative(sums[0][neuron]) * sum
                
        # Update the weights in the network
        # Update input weights
        for input in range(self.num_inputs + 1):
            for neuron in range(self.neurons_per_layer):
                self.weights[0][input, neuron] += activations[0][neuron] * deltas[0][neuron] # I think there's something wrong with this line
        # Update hidden layer weights
        for layer in range(1, self.num_hidden_layers):
            for prev_neuron in range(self.neurons_per_layer):
                for next_neuron in range(self.neurons_per_layer):
                    self.weights[layer][prev_neuron, next_neuron] += activations[layer][next_neuron] * deltas[layer][next_neuron] # and this
                    
        # Update output weights
        for output in range(self.num_outputs):
            for prev_neuron in range(self.neurons_per_layer):
                self.weights[self.num_hidden_layers][prev_neuron, output] += activations[self.num_hidden_layers][output] * deltas[self.num_hidden_layers][output] # and this
        
        # doesn't return anything (weights update in place)
            
            
        
    # Train a network by repeatedly backpropagating all the examples
    def train_network(self, num_examples, inputs_list, outputs_list, num_iterations):
        for i in range(num_iterations):
            for j in range(num_examples):
                self.back_propagate(inputs_list[j], outputs_list[j])
    
    # Convenience function for feeding forward then returning only the output
    def get_output(self, inputs):
        return self.feed_forward(inputs)[1][self.num_hidden_layers]
    

#### MAIN PROGRAM ##################################################################

# Test neural net by teaching it xor, this doesn't really work
# get_output on any of them gives a value that converges to

xor_net = NeuralNetwork(2, 1, 3, 1)
a = np.array([1.,1.])
b = np.array([1.,0.])
c = np.array([0.,1.])
d = np.array([0.,0.])
examples = [a, b, c, d]
labels = [np.array([0.]), np.array([1.]), np.array([1.]), np.array([0.])]

xor_net.train_network(4, examples, labels, 10000)
