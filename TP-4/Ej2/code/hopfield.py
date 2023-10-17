import numpy as np
from numpy import ndarray

from Ej2.code.letterParser import Letter, read_input_file


class Hopfield:

    def __init__(self, patterns: ndarray, num_iter):
        self.patterns = patterns
        self.W = self.train_weights()
        self.num_iter = num_iter

    def train_weights(self) -> ndarray:
        mat_k: ndarray = np.column_stack(self.patterns)

        to_mult = 1 / len(self.patterns[0])
        mat_w: ndarray = np.dot(mat_k, mat_k.T) * to_mult

        diag_w = np.diag(np.diag(mat_w))
        mat_w = mat_w - diag_w

        return mat_w

    def run(self, pattern: ndarray):
        s = pattern
        prev = pattern
        for i in range(self.num_iter):
            # Save data here
            # self.energy_df(i, self.energy(s))
            # self.pattern_df(i, s)

            s = np.sign(self.W @ s)

            if np.array_equal(s, prev):
                return s
            prev = s
        return s

    def energy(self, s):
        return -0.5 * s @ self.W @ s


if __name__ == "__main__":
    letras = read_input_file("../input/pattern_letters_2.json")
    hopfield = Hopfield(np.array(list(letras.values())), 1000)

    letter = letras["A"]

    noisy_letter = Letter.apply_noise(letter, 0.25, 1)

    pat = np.array(noisy_letter.data)
    s_final = hopfield.run(pat)

    print(Letter.print(letter))
    print(Letter.print(noisy_letter))
    print(Letter.print(s_final))

