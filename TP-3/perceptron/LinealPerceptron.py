from perceptron.SimplePerceptron import SimplePerceptron


class LinealPerceptron(SimplePerceptron):

    def __init__(self, weight_size, learning_rate: float, *args, **kwargs):
        super().__init__(weight_size, learning_rate, *args, **kwargs)
        self.epsilon = kwargs['epsilon']

    @staticmethod
    def activation(excitement: float) -> float:
        return excitement

    def get_tolerance(self):
        return self.epsilon

    def compute_delta_weights(self, expected_output, activation, current_input):
        act_der = self.activation_derivative()
        return self.learning_rate * (expected_output - activation) * act_der * current_input

    @staticmethod
    def activation_derivative() -> float:
        return 1

    def save_data(self, current_epoch, error):
        self.change = False
        self.data.save_data(weights=self.weights, error=error, current_epoch=current_epoch)


if __name__ == "__main__":
    perceptron = LinealPerceptron(3, 0.1, weights=[1.0, 1.0, 1.0])
    epoch = perceptron.train([[-1, -1], [-1, 1], [1, -1], [1, 1]], [-1, -1, -1, 1], 1000)

    print(f"Epoch: {epoch}")
