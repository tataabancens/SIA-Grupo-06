import numpy as np
import json

from numpy import ndarray


class Letter:
    @classmethod
    def print(cls, letter: ndarray):
        matriz_str = ""
        for i in range(5):
            for j in range(5):
                index = i * 5 + j
                if letter[index] == 1:
                    matriz_str += "* "
                else:
                    matriz_str += "  "
            matriz_str += '\n'
        return matriz_str

    @staticmethod
    def apply_noise(matriz: ndarray, noise_proportion: float, seed: int):
        """
        Aplica un ruido a la matriz cambiando aleatoriamente una proporción de bits.

        Args:
            matriz (numpy.ndarray): Matriz de entrada con valores 1 y -1.
            noise_proportion (float): Proporción de bits que serán cambiados (0.0 a 1.0).
            seed (int): Semilla para la generación de números aleatorios.
        Returns:
            numpy.ndarray: Matriz con el ruido aplicado.
        """
        if seed != 0:
            np.random.seed(seed)
        noisy_mat = np.copy(matriz)

        total_bits = matriz.size
        num_bits_to_shift = int(total_bits * noise_proportion)

        indexes_to_change = np.random.choice(total_bits, num_bits_to_shift, replace=False)

        noisy_mat[indexes_to_change] *= -1

        return noisy_mat


def read_input_file(filepath: str) -> dict[str, ndarray]:
    letters = {}

    with open(filepath, "r") as file:
        json_file = json.load(file)

        for letter in json_file:

            data: list[list[int]] = letter["data"]
            flat_data = np.array(data).flatten()
            letters[letter['name']] = flat_data
    return letters


if __name__ == "__main__":
    letras = read_input_file("../input/pattern_letters.json")

    for let in letras:
        print(Letter.print(letras[let]))

