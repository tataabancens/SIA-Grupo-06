import numpy as np
from perceptron.layer import Layer
from perceptron.optimizer import Optimizer, GradientDescent
from typing import List


class Dense(Layer):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        optimizer: Optimizer = GradientDescent(),
    ):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
        self.output_size = output_size
        self.optimizer = optimizer
        self.count = 0
        self.weights_accum = 0
        self.bias_accum = 0

    def forward(self, input: List[float]):
        self.input = input
        output = np.dot(self.weights, input) + self.bias
        return output

    def update(self):
        self.weights -= self.weights_accum / self.count
        self.bias -= self.bias_accum / self.count
        self.count = 0
        self.weights_accum = 0
        self.bias_accum = 0

    def backward(self, output_gradient: List[float], learning_rate: float):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        #
        # if update:
        #     self.weights -= self.optimizer.adjust(learning_rate, weights_gradient)
        #     self.bias -= self.optimizer.adjust(learning_rate, output_gradient)
        # else:
        self.weights_accum += self.optimizer.adjust(learning_rate, weights_gradient)
        self.bias_accum += self.optimizer.adjust(learning_rate, output_gradient)
        self.count += 1
        # self.weights -= learning_rate * weights_gradient
        # self.bias -= learning_rate * output_gradient
        return input_gradient
