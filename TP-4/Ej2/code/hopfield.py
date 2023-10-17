import numpy as np
from numpy import ndarray

from Ej2.code.letterParser import Letter, read_input_file


class Hopfield:

    def __init__(self, letters: list[Letter], num_iter):
        self.letters = letters
        self.W = self.train_weights()
        self.num_iter = num_iter

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
    hopfield = Hopfield(letras, 1000)

    letter = letras[2]

    noisy_letter = letter.copy_with_noise(0.25, 1)

    pat = np.array(noisy_letter.data)
    s_final = hopfield.run(pat)

    final_letter = Letter("j", s_final)
    print(letter)
    print(noisy_letter)
    print(final_letter)

