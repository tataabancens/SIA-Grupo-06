import numpy as np

from parse_letters import get_letters, print_letter, noisify
from perceptron.Autoencoder import Autoencoder
from perceptron.activation_functions import Tanh
from perceptron.errors import MeanSquared
from perceptron.optimizer import Adam
from perceptron.trainer import Batch

def latent_space(autoencoder: Autoencoder):
    train_x = get_letters()
    labels = ['`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '[', '|', ']', '~', 'DEL']
    for idx, input in train_x:
        coords = autoencoder.latent_space(input)
        print(coords)


def main():
    seed_value = 42
    train_x = get_letters()
    train_y = get_letters()

    for learning_rate in [0.0001]:
        np.random.seed(seed_value)
        p = Autoencoder([25, 15, 10], 35, 2, Tanh, Adam())
        p.train(MeanSquared, train_x, train_y, Batch(), 20000, learning_rate, False)

    for val in train_x:
        print_letter(p.predict_reshaped(val))
        print_letter(val)
        print_letter([1 if a >= 0.5 else 0 for a in p.predict_reshaped(val)])

    # print_letter(noisify(train_x[1], 0.2))



if __name__ == "__main__":
    main()
