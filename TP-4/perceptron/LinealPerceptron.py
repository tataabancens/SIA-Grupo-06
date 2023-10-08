import math

import numpy as np

from Ej2.config.config import Dataset
from perceptron.SimplePerceptron import SimplePerceptron


class LinealPerceptron(SimplePerceptron):
    def __init__(self, weight_size, learning_rate: float, *args, **kwargs):
        super().__init__(weight_size, learning_rate, *args, **kwargs)
        self.epsilon = kwargs["epsilon"]
        self.dataset: Dataset | None = None

    def activation(self, excitement: float) -> float:
        return excitement

    def get_tolerance(self):
        return self.epsilon

    def compute_delta_weights(
        self, expected_output, activation, current_input, excitement
    ):
        act_der = self.activation_derivative(excitement)
        return (
            self.learning_rate
            * (expected_output - activation)
            * act_der
            * current_input
        )

    @staticmethod
    def activation_derivative(activation: float) -> float:
        return 1

    def save_data(self, current_epoch, error, **kwargs):
        gen_error = 0
        try:
            gen_error = kwargs["gen_error"]
        except KeyError:
            pass
        self.change = False
        self.data.save_data(
            weights=self.weights,
            error=error,
            current_epoch=current_epoch,
            gen_error=gen_error,
        )

    @staticmethod
    def scale(numbers: List[float]):
        min_val = min(numbers)
        max_val = max(numbers)

        scaled_list = [2 * ((x - min_val) / (max_val - min_val)) - 1 for x in numbers]
        return scaled_list

    @staticmethod
    def inverse_scale(x, min_val, max_val):
        return (x + 1) * (max_val - min_val) / 2 + min_val

    def train_and_test(self, data: Dataset, epoch_limit: int, dataset_divider):
        current_epoch = 0
        min_error = np.finfo(np.float64).max
        best_w = np.array(self.weights)
        self.dataset = data

        train_data, test_data = dataset_divider(self.dataset)

        train_data.outputs = self.scale(train_data.outputs)
        test_data.outputs = self.scale(test_data.outputs)

        while min_error > self.get_tolerance() and current_epoch < epoch_limit:
            current_input, expected_output = self.random_np_input_from_list(
                train_data.inputs, train_data.outputs
            )

            excitement = self.excitement(current_input)
            activation = self.activation(excitement)

            self.weights += self.compute_delta_weights(
                expected_output, activation, current_input, excitement
            )

            if activation != expected_output:
                self.change = True

            error = self.compute_error(train_data.inputs, train_data.outputs)
            if error < min_error:
                min_error = error
                best_w = np.array(self.weights)

            current_epoch += 1
            generalizing_error = self.compute_error(test_data.inputs, test_data.outputs)
            self.save_data(current_epoch, error, gen_error=generalizing_error)
        return current_epoch, best_w, min_error

    def compute_error(self, data: list[List[float]], expected_outputs: List[float]):
        accum = 0
        for index, cur in enumerate(data):
            current_output = expected_outputs[index]
            current_input = self.add_x0_to_input(cur)

            excitement = self.excitement(current_input)
            activation = self.activation(excitement)

            inverse_scaled_act = self.inverse_scale(
                activation, self.dataset.min_output, self.dataset.max_output
            )
            inverse_scaled_cur = self.inverse_scale(
                current_output, self.dataset.min_output, self.dataset.max_output
            )
            accum += math.fabs(inverse_scaled_act - inverse_scaled_cur)
        return accum

    def random_np_input_from_list(
        self, data: list[List[float]], expected_outputs: List[float]
    ) -> tuple[np.ndarray, float]:
        random_index = np.random.randint(0, len(data))
        current_input = self.add_x0_to_input(data[random_index])

        expected_output = expected_outputs[random_index]
        return current_input, expected_output


if __name__ == "__main__":
    perceptron = LinealPerceptron(3, 0.1, weights=[1.0, 1.0, 1.0])
    epoch = perceptron.train(
        [[-1, -1], [-1, 1], [1, -1], [1, 1]], [-1, -1, -1, 1], 1000
    )

    print(f"Epoch: {epoch}")
