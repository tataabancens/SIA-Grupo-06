from parse_numbers import parse_numbers, numbers_map
from perceptron.MultiLayerPerceptron import MultiLayerPerceptron
from perceptron.activation_functions import Sigmoid
from perceptron.errors import MeanSquared
from perceptron.trainer import Batch
def main():
    x_train = parse_numbers()
    y_train = [(1 if i % 2 == 0 else 0) for i in range(10)]
    x_train_new = x_train[0:8]
    y_train_new = y_train[0:8]
    num_map = numbers_map()
    p = MultiLayerPerceptron([4,4], 7*5, 1, Sigmoid)
    print(p.predict(num_map[9]))
    p.train(MeanSquared, x_train_new, y_train_new, Batch(), 40000, 0.01, False)
    print(p.predict(num_map[9]))


if __name__ == "__main__":
    main()