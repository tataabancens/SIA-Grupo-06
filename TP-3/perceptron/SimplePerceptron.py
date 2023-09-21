import math
import numpy as np
import pandas as pd
from perceptron.dataClasses import DataClass, Ej1DataClass


class SimplePerceptron:

    def __init__(self, weight_size, learning_rate: float, data: DataClass, *args, **kwargs):
        self.learning_rate = learning_rate
        try:
            self.weights = np.array(kwargs['weights'])
        except KeyError:
            self.weights = np.random.uniform(-1, 1, size=weight_size)

        self.data: DataClass = data
        self.change = False

    def save_data(self):
        if len(self.weights) != 3:
            raise Exception("Weights size must be 3 for printing data in this SimplePerceptron implementation")
        if not self.change:
            return
        self.change = False
        self.data.save_data(weights=self.weights)

    def train(self, data: list[list[float]], expected_outputs: list[float], epoch_limit: int):
        current_epoch = 0
        min_error = np.finfo(np.float64).max
        best_w = np.array(self.weights)

        while min_error > self.get_tolerance() and current_epoch < epoch_limit:
            current_input, expected_output = self.random_np_input_from_list(data, expected_outputs)

            excitement = self.excitement(current_input)
            activation = self.activation(excitement)

            self.weights += self.compute_delta_weights(expected_output, activation, current_input)

            if activation != expected_output:
                self.change = True

            error = self.compute_error(data, expected_outputs)
            if error < min_error:
                min_error = error
                best_w = np.array(self.weights)

            current_epoch += 1
            self.save_data()
        return current_epoch, best_w

    def compute_delta_weights(self, expected_output, activation, current_input):
        return self.learning_rate * (expected_output - activation) * current_input

    @staticmethod
    def get_tolerance():
        return 0

    def excitement(self, input_array: np.ndarray) -> float:
        return float(np.dot(input_array, self.weights))

    @staticmethod
    def activation(excitement: float) -> int:
        return 1 if excitement > 0 else -1

    def calculate(self, cur_input: list[float]):
        current_input = self.add_x0_to_input(cur_input)

        excitement = self.excitement(current_input)
        return self.activation(excitement)

    def compute_error(self, data: list[list[float]], expected_outputs: list[float]):
        accum = 0
        for index, cur in enumerate(data):
            current_output = expected_outputs[index]
            current_input = self.add_x0_to_input(cur)

            excitement = self.excitement(current_input)
            activation = self.activation(excitement)

            accum += math.fabs(activation - current_output)
        return accum

    def random_np_input_from_list(self, data: list[list[float]],
                                  expected_outputs: list[float]) -> tuple[np.ndarray, float]:
        random_index = np.random.randint(0, len(data))
        current_input = self.add_x0_to_input(data[random_index])

        expected_output = expected_outputs[random_index]
        return current_input, expected_output

    @staticmethod
    def add_x0_to_input(cur_input: list[float]) -> np.ndarray:
        to_ret = np.array(cur_input)
        return np.insert(to_ret, 0, 1)

    def __repr__(self):
        return f"Weights: {self.weights}"


if __name__ == "__main__":
    perceptron = SimplePerceptron(3, 0.01)
    epoch = perceptron.train([[-1, -1], [-1, 1], [1, -1], [1, 1]], [-1, -1, -1, 1], 1000)
    print(perceptron.calculate([-1, -1]))


