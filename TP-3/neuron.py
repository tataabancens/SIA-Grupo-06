import sys
import numpy as np


class Neuron:

    def __init__(self, weights: list[float], learning_rate: float):
        """Initialize the neuron with the given umbral, weights and learning rate"""

        self.weights = np.empty(len(weights) + 1)
        self.weights[0] = np.random.default_rng(
        ).uniform(-1, 1)  # El umbral es un peso más

        for i, weight in enumerate(weights):
            np.append(self.weights, weight)

        self.learning_rate = learning_rate

    def exitation(self, inputs: list[float]):
        """Compute the exitation of the neuron"""
        return np.dot(inputs, self.weights)

    def activation(self, exitation: float):
        """Activation function of the neuron"""
        return 1 if exitation >= 0 else -1

    def process(self, inputs:  list[float]):
        """Process the inputs and return the output of the neuron"""
        input_with_umbral = np.insert(inputs, 0, 1)
        exitation = self.exitation(input_with_umbral)
        return self.activation(exitation)

    def delta_weight(self, expected_output: float, output: float,  inputs: list[float]):
        """Compute the delta weight of the neuron"""
        input_with_umbral = np.insert(inputs, 0, 1)
        delta_weight = np.empty(len(input_with_umbral))
        for i, input in enumerate(input_with_umbral):
            delta_weight[i] = self.learning_rate * input * \
                self.compute_error(expected_output, output)
        return delta_weight

    def compute_error(self, expected_output: float, output: float):
        """Compute the error between the expected output and the output of the neuron"""
        return expected_output - output

    def update_weights(self, weights_correction: list[float]):
        """Update the weights of the neuron"""
        for i in range(len(self.weights)):
            self.weights[i] += weights_correction[i]

    def compute_accuracy(self, current_output: list[bool]):
        """Compute the accuracy of the neuron"""
        return np.count_nonzero(current_output) / len(current_output)

    def train(self, inputs: list[list[float]], expected_output: list[float], epochs: int):
        """Train the neuron with the given inputs and expected outputs"""
        current_epoch = 0

        correct_output = np.zeros(len(inputs))
        error = np.ones(len(inputs))

        finished = False

        while current_epoch < epochs and not finished:
            for i, input in enumerate(inputs):

                output = self.process(input)

                error[i] = expected_output[i] - output

                correct_output[i] = 1 if error[i] == 0 else 0

                weights_correction = self.delta_weight(
                    expected_output[i], output, input)

                self.update_weights(weights_correction)

                if self.compute_accuracy(correct_output) == 1 or np.all(error == 0):
                    finished = True

            current_epoch += 1

        print("Epochs: " + str(current_epoch))
        print("Weights: " + str(self.weights))
