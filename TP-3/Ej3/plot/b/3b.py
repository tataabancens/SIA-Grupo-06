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

def plot_lines_error_comp(path):
    with open(path, 'r') as file:
        data = json.load(file)
    xs = [v["epoch"] for v in data["data"]]
    ys = [v["error"] for v in data["data"]]
    ys_test = [v["error_test"] for v in data["data"]]
    plt.errorbar(xs, ys_test, label="test set")
    plt.errorbar(xs, ys, label="training set")
    plt.legend()
    plt.title(f"{data['layers']}, p={data['proportion']}, {data['trainer']}, {data['activation']}, {data['optimizer']}, lr={data['learning_rate']}")
    plt.xlabel("epoch")
    plt.ylabel("error")
    plt.tight_layout()
    plt.show()

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


def ejc():
    seed_value = 42
    train_x = parse_numbers()
    train_y = []
    for i in range(10):
        arr = [0 for j in range(10)]
        arr[i] = 1
        train_y.append(arr)
    num_map = numbers_map()
    intensity = 0.1
    test_proportion = 0.5
    noisified_x = [noisify(num_map[i], intensity) for i in range(len(train_y))]
    train_x += noisified_x
    train_y += train_y
    for learning_rate in [0.01,0.001,0.0001]:
        np.random.seed(seed_value)
        p = MultiLayerPerceptron([10], 35, 10, Tanh, Adam())

        p.train(MeanSquared, train_x, train_y, Batch(), 2000, learning_rate,True, test_proportion, True)
    for learning_rate in [0.01,0.001,0.0001]:
        np.random.seed(seed_value)
        p = MultiLayerPerceptron([10], 35, 10, Tanh, GradientDescent())
        p.train(MeanSquared, train_x, train_y, Batch(), 2000, learning_rate,True, test_proportion, True)

    values = set()
    for filename in os.listdir(os.getcwd()):
        if 'json' in filename:
            values.add(filename.split('_')[0])

    for hash in values:
        plot_lines(hash)

    plot_lines_error_comp("563a81b6889ab57f82d2906ecd850baa5ba83d48dadee6a3f7d8935e790f0e77_20230924201114831.json")

def ejb():
    seed_value = 42
       # EJ 3.B
    train_x = [[0, 0], [0, 1], [1, 0], [1, 1]]
    train_y = [[0], [1], [1], [0]]
    test_proportion = 0.5
    for learning_rate in [0.01,0.001,0.0001]:
        np.random.seed(seed_value)
        p = MultiLayerPerceptron([4], 2, 1, Tanh, Adam())

        p.train(MeanSquared, train_x, train_y, Batch(), 2000, learning_rate,True, test_proportion, True)
    for learning_rate in [0.01,0.001,0.0001]:
        np.random.seed(seed_value)
        p = MultiLayerPerceptron([4], 2, 1, Tanh, GradientDescent())
        p.train(MeanSquared, train_x, train_y, Batch(), 2000, learning_rate,True, test_proportion, True)

    values = set()
    for filename in os.listdir(os.getcwd()):
        if 'json' in filename:
            values.add(filename.split('_')[0])

    for hash in values:
        plot_lines(hash)

    plot_lines_error_comp("56c86a38aefb9ac72bde7a13603213d4bf8620a2320678fae31724d2d66b35a5_20230924174337414.json")


def main():
    ejc()
    # EJ 3.B
    # train_x = [[0, 0], [0, 1], [1, 0], [1, 1]]
    # train_y = [[0], [1], [1], [0]]

if __name__ == '__main__':
    main()
