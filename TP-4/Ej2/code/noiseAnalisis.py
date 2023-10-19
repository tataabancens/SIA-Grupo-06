import pandas as pd
from numpy import ndarray

from Ej2.code.hopfield import Hopfield
from Ej2.code.letterParser import read_input_file, Letter
import numpy as np
from Ej2.config.config import load_config


def is_initial_pattern(prediction: ndarray, patterns: list[ndarray]):
    for pat in patterns:
        if np.array_equal(pat, prediction):
            return True
    return False


def main():
    config = load_config("../config/noiseConfig.json")
    all_letras = read_input_file(config.letters_path)

    letras = {letra: all_letras[letra] for letra in config.letters if letra in all_letras}

    hopfield = Hopfield(np.array(list(letras.values())), 5000)
    accuracy_list = []
    spurious = []

    for noise in config.noises:
        for letter, letter_value in letras.items():
            correct_amount = 0
            for i in range(config.n):
                noisy_letter = Letter.apply_noise(letter_value, noise, i)
                found_pat, prediction = hopfield.predict(letter_value, noisy_letter)
                if found_pat:
                    correct_amount += 1

                if not is_initial_pattern(prediction, list(letras.values())):
                    print(Letter.print(prediction))

            accuracy_list.append((noise, letter, correct_amount / config.n))

    df = pd.DataFrame(accuracy_list, columns=["noise", "letter", "accuracy"])
    print(df)



    # print(found_pat)
    # print(Letter.print(letter))
    # print(Letter.print(noisy_letter))
    # print(Letter.print(prediction))


if __name__ == "__main__":
    main()
