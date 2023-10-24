import pandas as pd
from numpy import ndarray

from Ej2.code.hopfield import Hopfield
from Ej2.code.letterParser import read_input_file, Letter
import numpy as np
from Ej2.config.config import load_config

import plotly.express as px
import plotly.graph_objects as go


def is_in_pattern_list(prediction: ndarray, patterns: list[ndarray]):
    for pat in patterns:
        if np.array_equal(pat, prediction):
            return True
    return False


def extract_and_save_accuracy():
    config = load_config("../config/noiseConfig.json")
    all_letras = read_input_file(config.letters_path)

    letras = {letra: all_letras[letra] for letra in config.letters if letra in all_letras}

    hopfield = Hopfield(np.array(list(letras.values())), 5000)
    accuracy_list = []
    spurious = []
    cycling = []

    spurious_data = []
    cycling_data = []

    for noise in config.noises:
        print(f"Starting noise {noise}")
        for letter, letter_value in letras.items():
            correct_amount = 0
            for i in range(config.n):
                noisy_letter = Letter.apply_noise(letter_value, noise, i)
                found_pat, prediction = hopfield.predict(letter_value, noisy_letter)
                if found_pat:
                    correct_amount += 1

                # Aca se ven estados espurios
                if not is_in_pattern_list(prediction, list(letras.values())):
                    if not is_in_pattern_list(prediction, spurious) and not is_in_pattern_list(prediction, cycling):
                        if hopfield.end_iter != hopfield.num_iter - 1:
                            spurious.append(prediction)
                            spurious_data.append((letter_value, noisy_letter, prediction))
                        else:
                            cycling.append(prediction)
                            cycling_data.append((letter_value, noisy_letter, prediction))

            accuracy_list.append((noise, letter, correct_amount / config.n))

    df = pd.DataFrame(accuracy_list, columns=["noise", "letter", "accuracy"])
    df.to_csv(f"../output/noises_{list(letras.keys())}.csv", index=False)


def plot_bar(filename: str):
    data_frame = pd.read_csv(filename)

    avg_accuracy = data_frame.groupby('noise')['accuracy'].mean().tolist()
    unique_noises = data_frame['noise'].unique().tolist()
    letters = data_frame['letter'].unique().tolist()

    fig = px.bar(data_frame, x=unique_noises, y=avg_accuracy, title=f"Model accuracy for 1000 tries {letters}")
    fig.update_yaxes(title_text="Accuracy")
    fig.update_xaxes(title_text="Noise proportion")
    fig.show()


def plot_bar_group(filename: str):
    data_frame = pd.read_csv(filename)
    unique_noises = data_frame['noise'].unique().tolist()
    letters = data_frame['letter'].unique().tolist()

    bars = []

    for letter in letters:
        filtered_data = data_frame[data_frame['letter'] == letter]
        accuracy_values = filtered_data['accuracy'].tolist()

        bars.append(go.Bar(name=letter, x=unique_noises, y=accuracy_values))

    fig = go.Figure(data=bars)
    # Change the bar mode

    fig.update_layout(title_text=f"Model accuracy for 1000 tries {letters}")
    fig.update_yaxes(title_text="Accuracy")
    fig.update_xaxes(title_text="Noise proportion")
    fig.update_layout(barmode='group')

    fig.show()


def plot_bar_group_noises(filename_list: list[str]):
    data_frame = pd.read_csv(filename_list[0])
    unique_noises = data_frame['noise'].unique().tolist()

    accuracy_list = []
    letter_sets = []
    for filename in filename_list:
        data_frame = pd.read_csv(filename)
        accuracy_list.append(data_frame.groupby('noise')['accuracy'].mean().tolist())
        letters = data_frame['letter'].unique().tolist()
        letter_sets.append(letters)

    bars = []

    for i, letter_set in enumerate(letter_sets):
        bars.append(go.Bar(name=f"{letter_set}", x=unique_noises, y=accuracy_list[i]))

    fig = go.Figure(data=bars)
    # Change the bar mode
    fig.update_layout(barmode='group')
    fig.update_layout(title_text=f"Model accuracy for 1000 tries")
    fig.update_yaxes(title_text="Accuracy")
    fig.update_xaxes(title_text="Noise proportion")
    fig.show()


if __name__ == "__main__":
    # extract_and_save_accuracy()
    # plot_bar("../output/noises_['B', 'C', 'D', 'O'].csv")
    # plot_bar_group("../output/noises_['B', 'C', 'D', 'O'].csv")

    fil_list = [
        "../output/noises_['F', 'I', 'L', 'X'].csv",
        "../output/noises_['F', 'O', 'V', 'Z'].csv",\
    ]
    plot_bar_group_noises(fil_list)
