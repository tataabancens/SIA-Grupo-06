from typing import List
import os

from utils.file_utils import load_config
from utils.constants import OutputFilenames, DEFAULT_NOISE
from utils.file_utils import get_output_dir, ConfigData
import plotly.graph_objects as go
from pandas import pandas as pd
from utils.poke_utils import create_all_pokemons_base
from dataclasses import dataclass
from pandas import DataFrame
from statistics import mean, stdev
from math import sqrt
from src.pokemon import Pokemon
from src.catching import attempt_catch
from typing import Tuple


def catch_pokemon(pokemon: Pokemon, pokeball_type:str, n:int) -> Tuple[float,float]:
    """
    Tries to catch a pokemon with a specific pokeball n times
    :param pokemon: the pokemon to catch
    :param pokeball_type: the name of the pokeball ot use
    :param n: the amount of times to try catching the pokemon
    :return: (Tuple[float,float]) the catch rate with its error
    """
    catches = []
    for _ in range(n):
        caught, rate = attempt_catch(pokemon, pokeball_type, DEFAULT_NOISE)
        catches.append(1 if caught else 0)
    catch_mean = mean(catches)
    return catch_mean, stdev(catches, catch_mean) / sqrt(n)


def ej1b_data(config: ConfigData):
    print(config)
    pokemons = create_all_pokemons_base(
        config.pokemon_names,
        config.levels[0],
        config.status_effects[0],
        config.healths[0]
    )

    @dataclass
    class Ej1bEntry:
        pokeball: str
        pokemon: str
        avg_prob: float

    entries: List[Ej1bEntry] = []
    for pokeball in config.pokeballs:
        for pokemon in pokemons:
            poke_prob_avg, _ = catch_pokemon(pokemon, pokeball, config.iterations)
            entries.append(Ej1bEntry(pokeball, pokemon.name, poke_prob_avg))

    df = DataFrame(entries)
    df.to_csv(get_output_dir().joinpath(OutputFilenames.EJ1B.value))


def bar_group_plot(pokeballs: List[str], pkmons: List[str], pkmon_avg_probs):
    bars = []

    for pkmon in pkmons:
        bars.append(go.Bar(name=pkmon, x=pokeballs, y=pkmon_avg_probs[pkmon]))

    fig = go.Figure(data=bars)
    # Change the bar mode
    fig.update_layout(barmode='group')
    fig.show()


def ej1b_plot():
    data_frame = pd.read_csv(get_output_dir().joinpath(OutputFilenames.EJ1B.value))

    grouped_data = data_frame.groupby(['pokemon', 'pokeball'])['avg_prob'].mean().unstack()
    pokeball_list = grouped_data.columns.tolist()

    normalized_avg_probs = {}
    for pokemon in data_frame['pokemon'].unique():
        base_avg_prob = grouped_data.loc[pokemon, "pokeball"]
        normalized_avg_probs[pokemon] = [avg_prob / base_avg_prob for avg_prob in grouped_data.loc[pokemon].tolist()]

    bar_group_plot(pokeball_list, data_frame['pokemon'].unique(), normalized_avg_probs)


if __name__ == "__main__":
    output_path = get_output_dir()

    config = load_config("Ej1b.json")

    os.makedirs(output_path, exist_ok=True)  # create dir if not exists
    ej1b_data(config)
    ej1b_plot()
