from perceptron.activation_functions import Activation, Sigmoid, Tanh
from perceptron.dense import Dense
from typing import Type
from perceptron.errors import ErrorFunction, MeanSquared
import numpy as np
from perceptron.trainer import Trainer, Batch, MiniBatch, Online


class MultiLayerPerceptron:
    def __init__(self, layers: list[int], input_size: int, output_size: int, activation: Type[Activation]) -> None:
        layers = [input_size] + layers + [output_size]
        self.input_size = input_size
        self.output_size = output_size
        layer_list = []
        i = 0
        while i < (len(layers) - 1):
            counts = layers[i:i + 2]
            layer_list.append(Dense(counts[0], counts[1]))
            layer_list.append(activation())
            i += 1

        self.layers = layer_list

    def predict(self, input):
        output = np.array([input]).T if isinstance(input, (list)) else input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, error_func: ErrorFunction, x_train, y_train, training_method: Trainer, epochs=1000,
              learning_rate=0.01, verbose=True):
        x_train = np.reshape(x_train, (len(x_train), self.input_size, 1))
        y_train = np.reshape(y_train, (len(y_train), self.output_size, 1))
        for e, dataset in enumerate(training_method.iterator(x_train, y_train, epochs)):
            error = 0
            for x, y in dataset:
                output = self.predict(x)

                error += error_func.eval(y, output)

                # backward
                grad = error_func.eval_derivative(y, output)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)

            error /= len(x_train)
            if verbose:
                print(f"{e + 1}/{epochs}, error={error}")


def main():
    p = MultiLayerPerceptron([4], 2, 1, Sigmoid)
    print(p.predict([0, 1]))
    train_x = [[0, 0], [0, 1], [1, 0], [1, 1]]
    train_y = [[0], [1], [1], [0]]
    p.train(MeanSquared(), train_x, train_y, MiniBatch(2), 20000, 0.01, False)

    print(p.predict([1, 1]))


if __name__ == "__main__":
    main()
