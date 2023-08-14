import os
from dataclasses import dataclass
from typing import List
from pandas import DataFrame

from utils.constants import OutputFilenames
from utils.file_utils import get_output_dir, load_config
from utils.poke_utils import stress_pokeball, create_all_pokemons_base, stress_pokeball_2

import plotly.express as px
import pandas as pd


@dataclass
class Ej1aEntry:
    pokeball: str
    avg_prob: str
    error: str


def ej1a():
    pokemons = create_all_pokemons_base(
        config.pokemon_names,
        config.levels[0],
        config.status_effects[0],
        config.healths[0]
    )

    entries: List[Ej1aEntry] = []
    for pokeball in config.pokeballs:
        poke_prob_avg, poke_error = stress_pokeball_2(pokeball, pokemons, config.iterations)
        entries.append(Ej1aEntry(pokeball, poke_prob_avg, poke_error))

    df = DataFrame(entries)
    df.to_csv(output_path.joinpath(OutputFilenames.EJ1A.value))
    print(df)


def plot_ej1a():
    data_frame = pd.read_csv(output_path.joinpath(OutputFilenames.EJ1A.value))
    print(data_frame)
    fig = px.bar(data_frame, x='pokeball', y='avg_prob', error_y='error', title="Pokeball accuracy")
    fig.show()


if __name__ == "__main__":
    output_path = get_output_dir()

    config = load_config()

    os.makedirs(output_path, exist_ok=True)  # create dir if not exists
    ej1a()
    plot_ej1a()





