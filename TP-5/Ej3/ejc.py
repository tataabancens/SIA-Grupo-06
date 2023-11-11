from parse_numbers import parse_numbers, numbers_map
from perceptron.MultiLayerPerceptron import MultiLayerPerceptron
from perceptron.activation_functions import Sigmoid
from perceptron.errors import MeanSquared
from perceptron.trainer import Batch
from noise import print_number, noisify
import numpy as np
def main():
    x_train = parse_numbers()
    seed_value = 42
    np.random.seed(seed_value)
    y_train = []
    for i in range(10):
        arr = [0 for j in range(10)]
        arr[i] = 1
        y_train.append(arr)
    num_map = numbers_map()
    p = MultiLayerPerceptron([16,16], 7*5, 10, Sigmoid)
    p.train(MeanSquared, x_train, y_train, Batch(), 30000, 0.01, False)

    intensity = 0.3

    test_x = [noisify(num_map[i], intensity) for i in range(len(x_train))]
    test_x += [noisify(num_map[i], intensity) for i in range(len(x_train))]
    test_x += [noisify(num_map[i], intensity) for i in range(len(x_train))]


    correct = 0
    for idx,val in enumerate(test_x):
        guess = np.argmax(p.predict(val))
        print(guess)
        print_number(idx, val)
        if y_train[idx][guess] == 1:
            correct += 1

    accuracy = float(correct)/len(x_train)
    print('accuracy:')
    print(accuracy)



if __name__ == "__main__":
    main()

