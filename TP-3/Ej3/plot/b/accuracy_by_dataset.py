import json

from Ej3.partition import partition
import numpy as np
from Ej3.parse_numbers import parse_numbers, numbers_map
from perceptron.MultiLayerPerceptron import MultiLayerPerceptron
from perceptron.activation_functions import Sigmoid, Tanh
from perceptron.errors import MeanSquared
from perceptron.optimizer import Adam, GradientDescent
from perceptron.trainer import Batch
from Ej3.noise import print_number, noisify
from matplotlib import pyplot as plt

def run_simulations(train_proportions, iterations, opt_method):
    json_result = {
        "iterations": iterations,
        "values": [],
        "optimization": opt_method
    }
    x_train = parse_numbers()
    y_train = [(1 if i % 2 == 0 else 0) for i in range(10)]

    for train_proportion in train_proportions:
        print(f"training proportion {train_proportion}")
        """
        {"iters": 10,
            "values": [{"dataset p": 0.1, "accuracy": 0.2, "std": 0.05}, {"dataset p": 0.3, "accuracy": 0.8, "std": 0.02}]}
        """
        training_proportion_obj = {"training proportion ": train_proportion}
        accuracies = []
        for iter in range(iterations):
            train_x, train_y, test_x, test_y = partition(x_train, y_train, train_proportion)
            p = MultiLayerPerceptron([10], 7*5, 1, Sigmoid, Adam())
            p.train(MeanSquared, train_x, train_y, Batch(), 20000, 0.01, False)

            #noisified_x = [noisify(num_map[i],intensity) for i in range(len(x_train))]

            correct = 0
            for idx, val in enumerate(test_x):
                real_number = x_train.index(val)
                guess = p.predict(val)
                print(f"real number {real_number}, guess: {guess}")
                if (guess >= 0.5 and real_number % 2 == 0) or (guess < 0.5 and real_number % 2 == 1):
                    correct += 1

            accuracy = float(correct)/len(test_x)
            accuracies.append(accuracy)

        training_proportion_obj["std"] = np.std(accuracies)
        training_proportion_obj["accuracy"] = np.mean(accuracies)
        json_result["values"].append(training_proportion_obj)

    with open('../../result3bAccuracyVsTrainingProportion.json', 'w') as json_file:
        json.dump(json_result, json_file)

def plot_accuracy_by_proportion(opt_method):
    with open('../../result3bAccuracyVsTrainingProportion.json', 'r') as file:
        data = json.load(file)

    training_proportions = [item["training proportion "] for item in data["values"]]
    accuracies = [item["accuracy"] for item in data["values"]]

    plt.plot(training_proportions, accuracies, marker='o', linestyle='-')
    plt.xlabel('Proportion of Dataset used for Training')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs. Proportion of Dataset used for Training using {opt_method}', fontsize=10)
    plt.grid(True)
    plt.ylim(0.1, 1.0)

    plt.show()
def main():
    run_simulations([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 10, 'ADAM')
    plot_accuracy_by_proportion('ADAM')

if __name__ == '__main__':
    main()
