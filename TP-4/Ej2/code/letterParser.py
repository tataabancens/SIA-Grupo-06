import numpy as np
import json

from numpy import ndarray


class Letter:
    def __init__(self, name: str, data: ndarray):
        self.name = name
        self.data = data

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


def read_input_file(filepath: str) -> list[Letter]:
    letters = []

    with open(filepath, "r") as file:
        json_file = json.load(file)

        for letter in json_file:

            data: list[list[int]] = letter["data"]
            flat_data = np.array(data).flatten()
            letters.append(Letter(letter['name'], flat_data))

    return letters


if __name__ == "__main__":
    letras = read_input_file("../input/simpleExample.json")
    print(letras)
