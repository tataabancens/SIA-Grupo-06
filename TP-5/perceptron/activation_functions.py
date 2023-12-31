import numpy as np
from perceptron.layer import Layer
from perceptron.activation import Activation

class ReLU(Activation):
    def __str__(self):
        return "ReLU"
    def __init__(self):
        def tanh(x):
            return x*(x > 0)

        def tanh_prime(x):
            return 1*(x > 0)

        super().__init__(tanh, tanh_prime)

class Linear(Activation):
    def __str__(self):
        return "Linear"
    def __init__(self):
        def tanh(x):
            return x

        def tanh_prime(x):
            return np.ones(x.shape)

        super().__init__(tanh, tanh_prime)
class Tanh(Activation):
    def __str__(self):
        return "Tanh"
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)


class Sigmoid(Activation):
    def __str__(self):
        return "Sigmoid"
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)


class Softmax(Layer):
    def forward(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output
    def __str__(self):
        return "SoftMax"
    def backward(self, output_gradient, learning_rate):
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)
