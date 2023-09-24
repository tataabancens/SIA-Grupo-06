from perceptron.activation_functions import Activation, Sigmoid, Tanh
from perceptron.dense import Dense
from typing import Type
from perceptron.errors import ErrorFunction, MeanSquared
import numpy as np
from perceptron.trainer import Trainer, Batch, MiniBatch, Online
from perceptron.optimizer import Optimizer, GradientDescent, Adam
from datetime import datetime
import json
import os

class MultiLayerPerceptron:
    def __init__(self, layers: list[int], input_size: int, output_size: int, activation: Type[Activation],
                 optimizer: Optimizer = GradientDescent()) -> None:
        layers = [input_size] + layers + [output_size]
        self.input_size = input_size
        self.output_size = output_size
        layer_list = []
        i = 0
        while i < (len(layers) - 1):
            counts = layers[i:i + 2]
            layer_list.append(Dense(counts[0], counts[1], optimizer.get_one()))
            layer_list.append(activation())
            i += 1

        self.layers = layer_list

    def predict(self, input):
        output = np.array([input]).T if isinstance(input, (list)) else input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, error_func: ErrorFunction, x, y, training_method: Trainer, epochs=1000,
              learning_rate=0.01, verbose=True, training_proportion=1.0, output=False):
        training_qty = int(len(x) * training_proportion)
        if training_qty == len(x) and training_proportion < 1.0:
            training_qty -= 1
        testing_qty = len(x) - training_qty
        prev_printed = 0
        x_train = np.reshape(x[:training_qty], (training_qty, self.input_size, 1))
        y_train = np.reshape(y[:training_qty], (training_qty, self.output_size, 1))
        x_test = np.reshape(x[training_qty:], (testing_qty, self.input_size, 1)) if training_qty < len(x) else []
        y_test = np.reshape(y[training_qty:], (testing_qty, self.output_size, 1)) if training_qty < len(x) else []
        stats = {
            "proportion": training_proportion,
            "data": []
        }
        first_time = True
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

            error_test = 0
            if len(x_test) > 0:
                for x, y in zip(x_test, y_test):
                    output = self.predict(x)

                    error_test += error_func.eval(y, output)

                error_test /= len(x_test)

            for layer in reversed(self.layers):
                layer.update()

            if first_time or e - prev_printed == 100:
                stats["data"].append({
                    "error": error,
                    "error_test": error_test,
                    "epoch": e
                })
                first_time = False
                prev_printed = e if e - prev_printed == 100 else 0

            if verbose:
                print(f"{e + 1}/{epochs}, error={error}, error_test={error_test}")
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if output:
            with open(f'{os.getcwd()}/{timestamp}.json', 'w') as json_file:
                json.dump(stats, json_file)


def main():
    p = MultiLayerPerceptron([4], 2, 1, Sigmoid, Adam())
    print(p.predict([0, 0]))
    train_x = [[0, 0], [0, 1], [1, 0], [1, 1]]
    train_y = [[0], [1], [1], [0]]
    p.train(MeanSquared, train_x, train_y, Batch(), 10000, 0.1, False, 0.8, True)

    print(p.predict([0, 0]))
    print(p.predict([0, 1]))
    print(p.predict([1, 0]))
    print(p.predict([1,1]))


if __name__ == "__main__":
    main()
