import numpy as np

from parse_letters import get_letters, print_letter, noisify
from perceptron.Autoencoder import Autoencoder
from perceptron.activation_functions import Tanh, Sigmoid
from perceptron.errors import MeanSquared
from perceptron.optimizer import Adam
from perceptron.trainer import Batch


def ej_b2(autoencoder: Autoencoder):

    train_x = get_letters()
    for val in train_x:
        noisified = noisify(val) # Aplicamos un ruido nuevo
        print_letter(noisified) # Asi se ve con ruido
        print_letter(autoencoder.predict_reshaped(noisified)) # Asi lo devuelve el autoencoder
        # print_letter(val)


def main():
    seed_value = 42
    train_x = get_letters()

    for learning_rate in [0.00005]:
        np.random.seed(seed_value)
        p = Autoencoder([25, 15, 10, 5], 35, 4, Sigmoid, Adam())
        p.train(MeanSquared, train_x, Batch(), 500000, learning_rate, noisify)


    ej_b2(p)






if __name__ == "__main__":
    main()
