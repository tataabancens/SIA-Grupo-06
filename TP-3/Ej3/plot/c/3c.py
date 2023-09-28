import json

import numpy as np
from Ej3.parse_numbers import parse_numbers, numbers_map
from perceptron.MultiLayerPerceptron import MultiLayerPerceptron
from perceptron.activation_functions import Sigmoid, Tanh
from perceptron.optimizer import GradientDescent, Adam
from perceptron.errors import MeanSquared
from perceptron.trainer import Batch
from Ej3.noise import print_number, noisify
from matplotlib import pyplot as plt


def run_simulations(intensities, iterations, opt_method):

    seed_value = 42
    json_result = {
        "iterations": iterations,
        "values": [],
        "optimization": opt_method
    }
    x_train = parse_numbers()


    y_train = []
    for i in range(10):
        arr = [0 for j in range(10)]
        arr[i] = 1
        y_train.append(arr)
    num_map = numbers_map()


    for intensity in intensities:
        np.random.seed(seed_value)

        test_x = [noisify(num_map[i], intensity) for i in range(len(x_train))]
        test_x += [noisify(num_map[i], intensity) for i in range(len(x_train))]
        test_x += [noisify(num_map[i], intensity) for i in range(len(x_train))]
        test_y = y_train + y_train + y_train
        print(f"intensity {intensity}")
        """
        {"iters": 10,
            "values": [{"intensity": 0.1, "accuracy": 0.2, "std": 0.05}, {"intensity": 0.3, "accuracy": 0.8, "std": 0.02}]}
        """
        intensity_obj = {"intensity": intensity}
        accuracies = []
        for iter in range(iterations):
            p = MultiLayerPerceptron([10], 7*5, 10, Tanh, GradientDescent())
            p.train(MeanSquared, x_train, y_train, Batch(), 120000, 0.01, False, [], [], False)

            correct = 0
            for idx,val in enumerate(test_x):
                guess = np.argmax(p.predict(val))
                # print(guess)
                # print(test_y[idx])
                if test_y[idx][guess] == 1:
                    correct += 1

            accuracy = float(correct)/len(test_x)
            accuracies.append(accuracy)

        intensity_obj["std"] = np.std(accuracies)
        intensity_obj["accuracy"] = np.mean(accuracies)
        json_result["values"].append(intensity_obj)

    with open('result.json', 'w') as json_file:
        json.dump(json_result, json_file)


def plot_accuracy_noise(path):
    with open(path, 'r') as file:
        data = json.load(file)
    labels = []
    values = []
    std_devs = []
    iters = data["iterations"]
    for vals in data["values"]:
        if iters == 0:
            iters = len(vals)
        labels.append(vals["intensity"])
        values.append(vals["accuracy"])
        std_devs.append(vals["std"])
    # Create the bar chart with error bars
    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, alpha=0.5,
            ecolor="black", yerr=std_devs, capsize=5, width=0.1)
    plt.xticks(labels, rotation=20, fontsize=10)
    plt.ylabel("Accuracy")
    plt.xlabel("Noise intensity")
    plt.title(f" Accuracy / noise intensity using {data['optimization']} (in {iters} iterations)")
    plt.subplots_adjust(bottom=0.3)

    plt.show()

def run_simulations_print_numbers(intensities, iterations):
    x_train = parse_numbers()
    y_train = []
    for i in range(10):
        arr = [0 for j in range(10)]
        arr[i] = 1
        y_train.append(arr)
    num_map = numbers_map()
    for intensity in intensities:
        for iter in range(iterations):
            p = MultiLayerPerceptron([10], 7 * 5, 10, Tanh)
            p.train(MeanSquared, x_train, y_train, Batch(), 30000, 0.01, False)

            noisified_x = [noisify(num_map[i], intensity) for i in range(len(x_train))]

            correct = 0
            for idx, val in enumerate(noisified_x):
                guess = np.argmax(p.predict(val))
                print(f"For the digit: {idx}, the perceptron guessed {guess}")
                print_number(idx, val)
                if y_train[idx][guess] == 1:
                    correct += 1





def main():
    intensities =  [0.1, 0.3, 0.5, 0.7]
    iterations = 4
    run_simulations(intensities, iterations, 'desc. gradient')
    plot_accuracy_noise("result.json")

    ##para imprimir los digitos con el guess:
    # run_simulations_print_numbers([0.3],1 )

if __name__ == '__main__':
    main()
