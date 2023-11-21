import numpy as np

from parse_letters import get_letters, print_letter, noisify
from perceptron.Autoencoder import Autoencoder
from perceptron.activation_functions import Tanh, Sigmoid
from perceptron.errors import MeanSquared
from perceptron.optimizer import Adam
from perceptron.trainer import Batch
import matplotlib.pyplot as plt
import seaborn as sns


def print_noisified_letters(intensity):
    train_x = get_letters()
    for val in train_x:
        noisified = noisify(val, intensity) # Aplicamos un ruido nuevo
        print_letter(noisified)

def ej_b2(autoencoder: Autoencoder):
    predicted = []
    noisified_letters = []
    train_x = get_letters()
    for val in train_x:
        noisified = noisify(val) # Aplicamos un ruido nuevo
        #print_letter(noisified) # Asi se ve con ruido
        noisified_letters.append(noisified)
        #print_letter(autoencoder.predict_reshaped(noisified)) # Asi lo devuelve el autoencoder
        predicted.append(autoencoder.predict_reshaped(noisified))
        # Agregar un grafico de letras correctas vs nivel de ruido
        # print_letter(val)

    print_letters_line(predicted, cmap='plasma')
    print_letters_line(noisified_letters, cmap='plasma')

def print_letters_line(letters, cmap='Blues', cmaps=[]):
    letts = np.array(letters)

    fig, ax = plt.subplots(1, len(letts))
    fig.set_dpi(360)

    if not cmaps:
        cmaps = [cmap] * len(letts)
    if len(cmaps) != len(letts):
        raise Exception('cmap list should be same length as letters')
    for i, subplot in enumerate(ax):
        create_letter_plot(letts[i].reshape(7, 5), ax=subplot, cmap=cmaps[i])
    plt.show()

def create_letter_plot(letter, ax, cmap='gray_r'):
    p = sns.heatmap(letter, ax=ax, annot=False, cbar=False, cmap=cmap, square=True, linewidth=0.5, linecolor='black')
    p.xaxis.set_visible(False)
    p.yaxis.set_visible(False)
    return p

def main():
    seed_value = 42
    train_x = get_letters()

    for learning_rate in [0.01]:
        np.random.seed(seed_value)
        p = Autoencoder([25, 15], 35, 4, Sigmoid, Adam())
        p.train(MeanSquared, train_x, Batch(), 50000, learning_rate, noisify)


    ej_b2(p)






if __name__ == "__main__":
    main()
