from perceptron.activation_functions import Activation, Sigmoid, Tanh, Linear
from perceptron.dense import Dense
from typing import Type
from perceptron.errors import ErrorFunction, MeanSquared
import numpy as np

from perceptron.reparam import Reparam
from perceptron.trainer import Trainer, Batch, MiniBatch, Online
from perceptron.optimizer import Optimizer, GradientDescent, Adam
from datetime import datetime
import json
import os
import hashlib
from parse_letters import get_letters, print_letter, noisify


class Autoencoder:
    def __init__(self, layers: list[int], input_size: int, output_size: int, activation: Type[Activation],
                 optimizer: Optimizer = GradientDescent(), variational: bool=False) -> None:
        self.latent_idx = len(layers) + 1
        layers = [input_size] + layers + [output_size] + [val for val in reversed(layers)] + [input_size]
        self.input_size = input_size
        self.output_size = input_size
        self.optimizer = optimizer
        self.activation = activation
        self.layer_config = layers
        self.variational = variational
        layer_list = []
        i = 0
        while i < (len(layers) - 1):
            counts = layers[i:i + 2]
            if i == self.latent_idx:
                self.reparam_layer = Reparam(counts[0], counts[1], optimizer.get_one()) if variational else Dense(counts[0], counts[1], optimizer.get_one())
                layer_list.append(self.reparam_layer)
                layer_list.append(Linear())
            else:
                layer_list.append(Dense(counts[0], counts[1], optimizer.get_one()))
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
        err = error_func.eval(pred, value)
        return err


    def generate(self, value):
        output = np.array([value]).T if isinstance(value, list) else value
        for layer in self.layers[self.latent_idx*2:]:
            output = layer.forward(output)
        return [item for sublist in output for item in sublist]

    def train(self, error_func: ErrorFunction, x_train, training_method: Trainer, epochs=1000,
              learning_rate=0.01, modifier_func=lambda x: x, output_file=False):

        idx = 0
        prev_printed = 0
        x_train = np.reshape(x_train, (len(x_train), self.input_size, 1))
        y_train = x_train

        stats = {
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
        prev_err = 1000000
        for e, dataset in enumerate(training_method.iterator(x_train, y_train, epochs)):
            error = 0
            # if idx % 10000 == 0:
            #     print(idx)
            for x, y in dataset:
                # print_letter(x)
                # print_letter(modifier_func(x))
                output = self.predict(modifier_func(x))
                error += error_func.eval(y, output)

                if self.variational:
                    error += self.reparam_layer.get_KL()


                # backward
                grad = error_func.eval_derivative(y, output)

                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)

            error /= len(x_train)




            for layer in reversed(self.layers):
                layer.update()

            if first_time or e - prev_printed == epochs / 100:
                stats["data"].append({
                    "error": error,
                    "epoch": e
                })
                first_time = False
                prev_printed = e if e - prev_printed == epochs / 100 else 0

            if idx == 0 or (idx+1) % 500 == 0:
                print(f"{e + 1}/{epochs}, error={error}, lr={learning_rate}")
            idx += 1

        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
        if output_file:
            with open(f'{os.getcwd()}/{hash_value}_{timestamp}.json', 'w') as json_file:
                json.dump(stats, json_file)
        return stats
