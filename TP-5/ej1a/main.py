import numpy as np
from matplotlib import pyplot as plt

from parse_letters import get_letters, print_letter, noisify, print_letters_line
from perceptron.Autoencoder import Autoencoder
from perceptron.activation_functions import Tanh, Sigmoid
from perceptron.errors import MeanSquared
from perceptron.optimizer import Adam
from perceptron.trainer import Batch

def ej_a3(autoencoder: Autoencoder):
    train_x = get_letters()
    labels = ['`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '[', '|', ']', '~']
    coordinates = []
    for input in train_x:
        c = autoencoder.latent_space(input)
        coordinates.append(c)

    x, y = zip(*coordinates)

    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y)

    # Add labels to the data points
    for i, label in enumerate(labels):
        plt.annotate(label, (x[i], y[i]), textcoords="offset points", xytext=(0,10), ha='center')

    # Set axis labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Scatter Plot with Labels')

    # Show the plot
    plt.grid(True)
    plt.show()

def ej_a2(autoencoder: Autoencoder):
    train_x = get_letters()
    predicted = []
    predicted_umbral = []

    for val in train_x:
        predicted.append(autoencoder.predict_reshaped(val))
        predicted_umbral.append([1 if a >= 0.5 else 0 for a in autoencoder.predict_reshaped(val)])

    print_letters_line(train_x, cmap='plasma')
    print_letters_line(predicted, cmap='plasma')
    print_letters_line(predicted_umbral, cmap='plasma')

def ej_a4(autoencoder: Autoencoder):
    train_x = get_letters()
    new_input = autoencoder.latent_space(train_x[1]) - 0.1 # Generamos un nuevo valor para el espacio latente a partir de la letra 'a'
    print_letter(autoencoder.generate(new_input))
    print_letter([1 if a >= 0.5 else 0 for a in autoencoder.generate(new_input)])

def main():
    seed_value = 42
    train_x = get_letters()

    for learning_rate in [0.0001]:
        np.random.seed(seed_value)
        p = Autoencoder([25, 15, 10, 5], 35, 2, Sigmoid, Adam())
        p.train(MeanSquared, train_x, Batch(), 10000, learning_rate)

    ej_a2(p)
    ej_a3(p)
    ej_a4(p)


if __name__ == "__main__":
    main()
