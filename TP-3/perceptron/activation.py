import numpy as np
from typing import Callable
from layer import Layer


class Activation(Layer):
    def __init__(self, activation: Callable[[float], float], activation_prime: Callable[[float], float]):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input: list[float]):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient: list[float], learning_rate: float):
        return np.multiply(output_gradient, self.activation_prime(self.input))
