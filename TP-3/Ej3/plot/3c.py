import numpy as np
from Ej3.parse_numbers import parse_numbers, numbers_map
from perceptron.MultiLayerPerceptron import MultiLayerPerceptron
from perceptron.activation_functions import Sigmoid
from perceptron.errors import MeanSquared
from perceptron.trainer import Batch
from Ej3.noise import print_number, noisify

def run_simulations(intensity, iterations):
    """
    {"iters": 10,
        "values": [{"intensity": 0.1, "accuracy": 0.2, "std": 0.05}, {"intensity": 0.3, "accuracy": 0.8, "std": 0.02}]}

    """
    x_train = parse_numbers()
    y_train = []
    for i in range(10):
        arr = [0 for j in range(10)]
        arr[i] = 1
        y_train.append(arr)
    num_map = numbers_map()
    for iter in range(iterations):
        p = MultiLayerPerceptron([16,16], 7*5, 10, Sigmoid)
        p.train(MeanSquared, x_train, y_train, Batch(), 30000, 0.01, False)

        noisified_x = [noisify(num_map[i],intensity) for i in range(len(x_train))]

        correct = 0
        for idx,val in enumerate(noisified_x):
            guess = np.argmax(p.predict(val))
            print(guess)
            print_number(val)
            if y_train[idx][guess] == 1:
                correct += 1

        accuracy = float(correct)/len(x_train)
        print('accuracy:')
        print(accuracy)

def main():
    print('hola')

if __name__ == '__main__':
    main()
