from perceptron.activation_functions import Activation, Sigmoid, Tanh, Linear
from perceptron.dense import Dense
from typing import Type
from perceptron.errors import ErrorFunction, MeanSquared
import numpy as np
from perceptron.trainer import Trainer, Batch, MiniBatch, Online
from perceptron.optimizer import Optimizer, GradientDescent, Adam
from datetime import datetime
import json
import os
import hashlib
from parse_letters import get_letters, print_letter


class Autoencoder:
    def __init__(self, layers: list[int], input_size: int, output_size: int, activation: Type[Activation],
                 optimizer: Optimizer = GradientDescent()) -> None:
        self.latent_idx = len(layers) + 1
        layers = [input_size] + layers + [output_size] + [val for val in reversed(layers)] + [input_size]
        self.input_size = input_size
        self.output_size = input_size
        self.optimizer = optimizer
        self.activation = activation
        self.layer_config = layers
        layer_list = []
        i = 0
        while i < (len(layers) - 1):
            counts = layers[i:i + 2]
            layer_list.append(Dense(counts[0], counts[1], optimizer.get_one()))
            if i == self.latent_idx:
                print(layers[i])
                layer_list.append(Linear())
            else:
                layer_list.append(activation())
            i += 1

        self.layers = layer_list

    def latent_space(self, input):
        output = np.array([input]).T if isinstance(input, (list)) else input
        for idx, layer in enumerate(self.layers):
            if idx == 2*self.latent_idx:
                return output
            output = layer.forward(output)



    def predict(self, input):
        output = np.array([input]).T if isinstance(input, (list)) else input
        for layer in self.layers:
            output = layer.forward(output)
        return output
    def predict_reshaped(self, input):
        value = self.predict(input)
        return [item for sublist in value for item in sublist]

    def error(self, value, error_func: ErrorFunction):
        pred = self.predict(value)
        return error_func.eval(pred, value)

    def train(self, error_func: ErrorFunction, x_train, y_train, training_method: Trainer, epochs=1000,
              learning_rate=0.01, verbose=True, x_test=None, y_test=None, output_file=False):
        if y_test is None:
            y_test = []
        if x_test is None:
            x_test = []
        # training_qty = int(len(x) * training_proportion)
        # if training_qty == len(x) and training_proportion < 1.0:
        #     training_qty -= 1
        # testing_qty = len(x) - training_qty
        # x_train = np.reshape(x[:training_qty], (training_qty, self.input_size, 1))
        # y_train = np.reshape(y[:training_qty], (training_qty, self.output_size, 1))
        # x_test = np.reshape(x[training_qty:], (testing_qty, self.input_size, 1)) if training_qty < len(x) else []
        # y_test = np.reshape(y[training_qty:], (testing_qty, self.output_size, 1)) if training_qty < len(x) else []
        prev_printed = 0
        training_proportion = len(x_train) / (len(x_train) + len(x_test))
        x_train = np.reshape(x_train, (len(x_train), self.input_size, 1))
        y_train = np.reshape(y_train, (len(y_train), self.output_size, 1))
        x_test = np.reshape(x_test, (len(x_test), self.input_size, 1))
        y_test = np.reshape(y_test, (len(y_test), self.output_size, 1))
        stats = {
            "proportion": training_proportion,
            "optimizer": self.optimizer.__str__(),
            "data": [],
            "trainer": training_method.__str__(),
            "activation": self.activation().__str__(),
            "layers": f"{self.layer_config}",
        }
        json_string = json.dumps(stats)
        stats["learning_rate"] = learning_rate

        # Step 2: Calculate the hash of the JSON string
        hash_obj = hashlib.sha256(json_string.encode())  # You can use other hash algorithms as needed

        # Step 3: Get the hexadecimal representation of the hash
        hash_value = hash_obj.hexdigest()
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

            if first_time or e - prev_printed == epochs / 100:
                stats["data"].append({
                    "error": error,
                    "error_test": error_test,
                    "epoch": e
                })
                first_time = False
                prev_printed = e if e - prev_printed == epochs / 100 else 0

            if verbose:
                print(f"{e + 1}/{epochs}, error={error}, error_test={error_test}")
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
        if output_file:
            with open(f'{os.getcwd()}/{hash_value}_{timestamp}.json', 'w') as json_file:
                json.dump(stats, json_file)
        return stats


def main():
    seed_value = 42
    train_x = get_letters()
    train_y = get_letters()

    print(train_x[0])
    for learning_rate in [0.0001]:
        np.random.seed(seed_value)
        p = Autoencoder([25, 15, 10], 35, 2, Tanh, Adam())
        p.train(MeanSquared, train_x, train_y, Batch(), 200000, learning_rate, False)

    print_letter([1 if val >= 0.5 else 0 for val in p.predict_reshaped(train_x[1])])
    print_letter(p.predict_reshaped(train_x[1]))
    print_letter(train_x[1])

    print_letter(p.predict_reshaped(train_x[3]))
    print_letter(train_x[3])
    print(p.error(train_x[1], MeanSquared))
    print(p.latent_space(train_x[1]))


if __name__ == "__main__":
    main()
