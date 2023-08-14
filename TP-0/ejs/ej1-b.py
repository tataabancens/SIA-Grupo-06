from typing import List
from pathlib import Path
import os
from utils.file_utils import load_config

from utils.constants import OUTPUT_PATH, OutputFilenames
from utils.file_utils import get_output_dir
import plotly.graph_objects as go
from pandas import pandas as pd
from utils.poke_utils import stress_pokeball, create_all_pokemons_base


def ej1b_data():
    pokemons = create_all_pokemons_base(
        config.pokemon_names,
        config.levels[0],
        config.status_effects[0],
        config.healths[0]
    )

    with open(output_path.joinpath(OutputFilenames.EJ1B.value), "w") as csv_f:
        csv_f.write("pokeball,pokemon,avg_prob,stdev\n")

        for pokeball in config.pokeballs:
            for pokemon in pokemons:
                poke_prob_avg, poke_stdev = stress_pokeball(pokeball, [pokemon], config.iterations)
                csv_f.write(f"{pokeball},{pokemon.name},{poke_prob_avg},{poke_stdev}\n")


def bar_group_plot(pokeballs: List[str], pkmons: List[str], pkmon_avg_probs):
    bars = []

    for pkmon in pkmons:
        bars.append(go.Bar(name=pkmon, x=pokeballs, y=pkmon_avg_probs[pkmon]))

    fig = go.Figure(data=bars)
    # Change the bar mode
    fig.update_layout(barmode='group')
    fig.show()


def ej1b_plot():
    data_frame = pd.read_csv(get_output_dir().joinpath(OutputFilenames.EJ1B.value), sep=',')

    grouped_data = data_frame.groupby(['pokemon', 'pokeball'])['avg_prob'].mean().unstack()
    pokeball_list = grouped_data.columns.tolist()

    normalized_avg_probs = {}
    for pokemon in data_frame['pokemon'].unique():
        base_avg_prob = grouped_data.loc[pokemon, "pokeball"]
        normalized_avg_probs[pokemon] = [avg_prob / base_avg_prob for avg_prob in grouped_data.loc[pokemon].tolist()]

    bar_group_plot(pokeball_list, data_frame['pokemon'].unique(), normalized_avg_probs)


if __name__ == "__main__":
    output_path = get_output_dir()

    config = load_config()

    os.makedirs(output_path, exist_ok=True)  # create dir if not exists
    ej1b_data()
    ej1b_plot()
