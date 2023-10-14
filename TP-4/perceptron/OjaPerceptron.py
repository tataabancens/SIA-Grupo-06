import math
import numpy as np
from typing import List

from perceptron.dataClasses import DataClass


class OjaPerceptron:
    def __init__(
        self, weight_size, learning_rate: float, data,*args, **kwargs
    ):
        self.learning_rate = learning_rate
        try:
            self.weights = np.array(kwargs["weights"])
        except KeyError:
            self.weights = np.random.uniform(-1, 1, size=weight_size)

        # self.data: DataClass = data
        self.change = False

    # def save_data(self, current_epoch, error):
    #     if len(self.weights) != 3:
    #         raise Exception(
    #             "Weights size must be 3 for printing data in this SimplePerceptron implementation"
    #         )
    #     if not self.change:
    #         return
    #     self.change = False
    #     self.data.save_data(weights=self.weights)

    def train(
        self, data: list[List[float]], epoch_limit: int
    ):
        current_epoch = 0
        best_w = np.array(self.weights)

        while  current_epoch < epoch_limit:
            current_input = self.random_np_input_from_list(
                data
            )

            excitement = self.excitement(current_input)
            activation = self.activation(excitement)

            self.weights += self.compute_delta_weights(
                self.weights, activation, current_input, excitement
            )

            if current_epoch >= epoch_limit:
                best_w = np.array(self.weights)

            current_epoch += 1
            # self.save_data(current_epoch)
        return current_epoch, best_w

    def compute_delta_weights(
        self, weights, activation, current_input, excitement
    ):
        return self.learning_rate * (activation*current_input - activation*activation*weights)

    @staticmethod
    def get_tolerance():
        return 0

    def excitement(self, input_array: np.ndarray) -> float:
        return float(np.dot(input_array, self.weights))

    @staticmethod
    def activation(excitement: float) -> int:
        return 1 if excitement > 0 else -1

    def calculate(self, cur_input: List[float]):
        current_input = self.add_x0_to_input(cur_input)

        excitement = self.excitement(current_input)
        return self.activation(excitement)

    # def compute_error(self, data: list[List[float]], expected_outputs: List[float]):
    #     accum = 0
    #     for index, cur in enumerate(data):
    #         current_output = expected_outputs[index]
    #         current_input = self.add_x0_to_input(cur)
    #
    #         excitement = self.excitement(current_input)
    #         activation = self.activation(excitement)
    #
    #         accum += math.fabs(activation - current_output)
    #     return accum

    def random_np_input_from_list(
        self, data: list[List[float]]
    ) -> tuple[np.ndarray, float]:
        random_index = np.random.randint(0, len(data))
        current_input = self.add_x0_to_input(data[random_index])

        return current_input

    @staticmethod
    def add_x0_to_input(cur_input: List[float]) -> np.ndarray:
        to_ret = np.array(cur_input)
        return np.insert(to_ret, 0, 1)

    def __repr__(self):
        return f"Weights: {self.weights}"


if __name__ == "__main__":
    perceptron = OjaPerceptron(3, 0.01,None)
    epoch = perceptron.train(
        [[-1, -1], [-1, 1], [1, -1], [1, 1]],  1000
    )
    print(perceptron.calculate([-1, -1]))
