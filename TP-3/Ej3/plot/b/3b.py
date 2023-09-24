import json

import numpy as np
from Ej3.parse_numbers import parse_numbers, numbers_map
from perceptron.MultiLayerPerceptron import MultiLayerPerceptron
from perceptron.activation_functions import Sigmoid, Tanh
from perceptron.errors import MeanSquared
from perceptron.trainer import Batch
from perceptron.optimizer import GradientDescent, Adam
from Ej3.noise import print_number, noisify
from matplotlib import pyplot as plt
import os

def plot_lines(hash):  # misma configs o sea mismo hash con dif fecha
    output_files = []
    identifier = hash
    for filename in os.listdir(os.getcwd()):
        file_path = os.path.join(os.getcwd(), filename)
        if os.path.isfile(file_path) and identifier in filename:
            output_files.append(file_path)

    # [csv, csv]
    for path in output_files:
        with open(path, 'r') as file:
            data = json.load(file)
        xs = [v["epoch"] for v in data["data"]]
        ys = [v["error"] for v in data["data"]]
        plt.errorbar(xs, ys, label=f"learning_rate={data['learning_rate']}")
        plt.legend()
    plt.title(f"{data['layers']}, p={data['proportion']}, {data['trainer']}, {data['activation']}, {data['optimizer']}")
    plt.xlabel("epoch")
    plt.ylabel("error")
    plt.tight_layout()
    plt.show()




def main():
    seed_value = 42
    train_x = [[0, 0], [0, 1], [1, 0], [1, 1]]
    train_y = [[0], [1], [1], [0]]
    # for learning_rate in [0.01,0.001,0.0001]:
    #     np.random.seed(seed_value)
    #     p = MultiLayerPerceptron([4], 2, 1, Tanh, Adam())
    #
    #     p.train(MeanSquared, train_x, train_y, Batch(), 2000, learning_rate,True, 0.5, True)
    # for learning_rate in [0.01,0.001,0.0001]:
    #     np.random.seed(seed_value)
    #     p = MultiLayerPerceptron([4], 2, 1, Tanh, GradientDescent())
    #     p.train(MeanSquared, train_x, train_y, Batch(), 2000, learning_rate,True, 0.5, True)

    values = set()
    for filename in os.listdir(os.getcwd()):
        if 'json' in filename:
            values.add(filename.split('_')[0])

    for hash in values:
        plot_lines(hash)

if __name__ == '__main__':
    main()
