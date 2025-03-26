import random
from engine import Value

"""
notes:
- each layer in an MLP has a number of neurons
- neurons in a layer are not connected to each other but they are all connected to the inputs
- MLP stands for multi layer perceptron
- in an MLP, layers feed into each other sequentially
"""

class Neuron:

    def __init__(self, number_of_inputs):
        # a weight that is a random number btw -1 and 1 for every one of those inputs
        self.w = [Value(random.uniform(-1,1)) for _ in range(number_of_inputs)]
        # a bias that controls the overall trigger happiness of this neuron
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        # this will be the forward pass of a single neuron
        # w * x + b where w * x is a dot product
        # zip takes 2 iterators and creates a new iterator
        # sum has a second optionl parameter to start adding on, default to 0; you could set this to self.b
        activation = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        # pass the activation through a nonlinearity
        out = activation.tanh()
        return out
    
    # we want some convenience code to gather up all the parameters of the neural net,
    # and we can nudge them a tiny amount based on the grad informaton
    def parameters(self):
        # self.w is a list; list + list gives you a list
        # btw, pyTorch has a parameters on every neural network module
        return self.w + [self.b]
    
class Layer:

    # a layer is a list of neurons
    # number_of_inputs - how many neurons they layer has
    # number_of_outputs - how many neurons do you want
    def __init__(self, number_of_inputs, number_of_outputs):
        self.neurons = [Neuron(number_of_inputs) for _ in range(number_of_outputs)]

    def __call__(self, x):
        outputs = [n(x) for n in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs
    
    def parameters(self):
        params = []
        for neuron in self.neurons:
            neuron_params = neuron.parameters()
            params.extend(neuron_params)
        return params

class MultiLayerPerceptron:
    
    # number_of_inputs - single value
    # list_of_number_of_outputs - list defining size of each layer in the MLP
    def __init__(self, number_of_inputs, list_of_number_of_outputs):
        sz = [number_of_inputs] + list_of_number_of_outputs
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(list_of_number_of_outputs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

if __name__ == "__main__":
    # x = [2.0, 3.0]
    # n = Neuron(2)
    # o = n(x)  # n(x) will use __call__()
    # print(o)

    # x = [2.0, 3.0]
    # n = Layer(2, 3)  # 2D neurons, 3 neurons per layer
    # print(n(x))

    x = [2.0, 3.0, -1.0]  # 3 dimensional input
    n = MultiLayerPerceptron(3, [4, 4, 1])  # 3 input and 2 layers of 4 and 1 output layer
    print(n(x))  # viola, forward pass of an MLP
    