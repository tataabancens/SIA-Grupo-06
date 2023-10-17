from Ej2.code.hopfield import Hopfield
from Ej2.code.letterParser import Letter, read_input_file
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import ndarray


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


if __name__ == "__main__":
    letras = read_input_file("../input/pattern_letters.json")
    only_letras = list(letras.values())

    sub_letras_list = [only_letras[i:i + 6] for i in range(0, len(only_letras), 6)]
    for sub_letras in sub_letras_list:
        print_letters_line(sub_letras)
