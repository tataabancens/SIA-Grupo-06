import numpy as np
from numpy import ndarray

from Ej2.code.letterParser import Letter, read_input_file


class Hopfield:

    def __init__(self, letters: list[Letter]):
        self.letters = letters
        self.W = self.train_weights()

    def train_weights(self) -> ndarray:
        patterns = []
        for letter in self.letters:
            patterns.append(letter.data)

        mat_k: ndarray = np.column_stack(patterns)

        to_mult = 1 / len(patterns[0])
        mat_w: ndarray = np.dot(mat_k, mat_k.T) * to_mult

        diag_w = np.diag(np.diag(mat_w))
        mat_w = mat_w - diag_w

        return mat_w


if __name__ == "__main__":
    letras = read_input_file("../input/simpleExample.json")
    hopfield = Hopfield(letras)

