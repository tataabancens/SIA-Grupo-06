import numpy as np
from numpy import ndarray

from letterParser import Letter, read_input_file


class Hopfield:

    def __init__(self, patterns: ndarray, num_iter):
        self.patterns = patterns
        self.W = self.train_weights()
        self.num_iter = num_iter
        self.end_iter = 0
        self.energies = []
        self.s_evolution = []

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
        self.s_evolution = []
        self.s_evolution.append(s.copy())  
        for i in range(self.num_iter):
            self.end_iter = i

            energy_val = self.energy(s)
            self.energies.append(energy_val)
            s = np.sign(self.W @ s)
            self.s_evolution.append(s.copy())  

            if np.array_equal(s, prev):
                return s, self.energies, self.s_evolution
            prev = s
        return s, self.energies, self.s_evolution

    def predict(self, true_pat: ndarray, noisy_pat: ndarray):
        prediction, energies, s_evolution = self.run(noisy_pat)
        return np.array_equal(prediction, true_pat), prediction, s_evolution

    def energy(self, s):
        return -0.5 * s @ self.W @ s 


if __name__ == "__main__":
    letras = read_input_file("../input/pattern_letters.json")
    hopfield = Hopfield(np.array(list(letras.values())), 1000)

    letter = letras["O"]

    noisy_letter = Letter.apply_noise(letter, 0.25, 1)

    pat = np.array(noisy_letter.data)
    s_final = hopfield.run(pat)

    print(Letter.print(letter))
    print(Letter.print(noisy_letter))
    print(Letter.print(s_final))
