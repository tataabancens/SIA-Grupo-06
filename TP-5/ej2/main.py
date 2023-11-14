import numpy as np
from matplotlib import pyplot as plt

from ej2.emoji import get_emoji_vectors, draw_emoji
from parse_letters import get_letters, print_letter, noisify
from perceptron.Autoencoder import Autoencoder
from perceptron.activation_functions import Tanh, Sigmoid
from perceptron.errors import MeanSquared
from perceptron.optimizer import Adam, GradientDescent
from perceptron.trainer import Batch

latent_size = 10

def ej_c(autoencoder: Autoencoder, train):
    for value in train:
        print(autoencoder.latent_space(value))
        draw_emoji(value)
        draw_emoji(autoencoder.predict_reshaped(value))
    input = (autoencoder.latent_space(train[0])*0.3 + autoencoder.latent_space(train[1])*0.7)/2
    output = autoencoder.generate(input)
    draw_emoji(output)

    # output = autoencoder.generate([0.8 for i in range(latent_size)])
    # draw_emoji(output)

def main():
    seed_value = 44
    train = get_emoji_vectors()
    train1 = [train[0], train[1]]
    # train2 = [train[0], train[2]]

    for train_x in [train]:
        learning_rate = 0.01
        np.random.seed(seed_value)
        p = Autoencoder([200, 100, 50], 24*24, latent_size, Sigmoid, Adam(), True) # Ver diferencia usando VAE y sin VAE (Pasando True o False)
        p.train(MeanSquared, train_x, Batch(), 100000, learning_rate)
        ej_c(p, train_x)






if __name__ == "__main__":
    main()
