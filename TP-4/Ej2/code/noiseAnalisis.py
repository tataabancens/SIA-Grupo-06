from Ej2.code.hopfield import Hopfield
from Ej2.code.letterParser import read_input_file, Letter
import numpy as np


def main():
    all_letras = read_input_file("../input/pattern_letters.json")
    letters = ("O", "P", "W", "Z")

    letras = {letra: all_letras[letra] for letra in letters if letra in all_letras}

    hopfield = Hopfield(np.array(list(letras.values())), 1000)

    letter = letras["W"]

    noisy_letter = Letter.apply_noise(letter, 0.2, 1)
    found_pat, prediction = hopfield.predict(letter, noisy_letter)

    print(found_pat)
    print(Letter.print(letter))
    print(Letter.print(noisy_letter))
    print(Letter.print(prediction))


if __name__ == "__main__":
    main()
