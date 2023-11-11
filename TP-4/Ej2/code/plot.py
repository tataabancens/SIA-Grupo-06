from hopfield import Hopfield
from letterParser import Letter, read_input_file
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import ndarray
import numpy as np


def create_letter_plot(letter, ax, cmap='Blues'):
    p = sns.heatmap(letter, ax=ax, annot=False, cbar=False, cmap=cmap, square=True, linewidth=2, linecolor='black')
    p.xaxis.set_visible(False)
    p.yaxis.set_visible(False)
    return p


def print_letters_line(letters: list[ndarray], cmap='Blues', cmaps=[]):
    fig, ax = plt.subplots(1, len(letters))
    fig.set_dpi(360)

    if not cmaps:
        cmaps = [cmap] * len(letters)
    if len(cmaps) != len(letters):
        raise Exception('cmap list should be same length as letters')
    for i, subplot in enumerate(ax):
        create_letter_plot(letters[i].reshape(5, 5), ax=subplot, cmap=cmaps[i])
    plt.show()


def plot_energies(energies):
        plt.plot(range(len(energies)), energies, marker='o')
        plt.title('Energía de las iteraciones')
        plt.xlabel('Iteraciones')
        plt.ylabel('Energía')
        plt.show()


def plot_s_evolution(s_evolution):
        num_plots = len(s_evolution)
        num_rows = num_plots // 3 + 1 if num_plots % 3 != 0 else num_plots // 3
        plt.figure(figsize=(15, 5 * num_rows))

        for i, s in enumerate(s_evolution):
            plt.subplot(num_rows, 3, i + 1)
            plt.imshow(np.array(s).reshape((int(np.sqrt(len(s))), -1)), cmap='Blues', interpolation='nearest')
            plt.title(f"Iteration {i}")
            plt.axis('off')
        plt.show()


def impresionLetras():
    letras = read_input_file("../input/pattern_letters.json")
    only_letras = list(letras.values())

    sub_letras_list = [only_letras[i:i + 6] for i in range(0, len(only_letras), 6)]
    for sub_letras in sub_letras_list:
        print_letters_line(sub_letras)


def impresionrun():
    all_letras = read_input_file("../input/pattern_letters.json")
    letters = ("A", "P", "W", "Z")

    letras = {letra: all_letras[letra] for letra in letters if letra in all_letras}

    hopfield = Hopfield(np.array(list(letras.values())), 1000)

    letter = letras["A"]

    noisy_letter = Letter.apply_noise(letter, 0.2, 326)
    pat = np.array(noisy_letter.data)
    s, energies, s_evolution = hopfield.run(pat)

    for i in range(len(s_evolution)):
        s_evolution[i] = np.where(s_evolution[i] == 0, -1, s_evolution[i])

    # plot_energies(energies)
    # plot_s_evolution(s_evolution)

    sub_letras_list = [s_evolution[i:i + 4] for i in range(0, len(s_evolution), 4)]
    for sub_letras in sub_letras_list:
        print_letters_line(sub_letras)
    # print_letters_line(s_evolution)


if __name__ == "__main__":
    # impresionLetras()
    impresionrun()


