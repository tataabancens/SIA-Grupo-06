
import json
import math

from typing import List, Tuple
import sys
from pathlib import Path
import os
from statistics import mean, stdev

from src.pokemon import PokemonFactory, StatusEffect, Pokemon
from utils.constants import OUTPUT_PATH, DEFAULT_NOISE
from ejs.ej1b import ej1b_data, ej1b_plot
from utils.file_utils import ConfigData
from utils.poke_utils import stress_pokeball

def write_CSV(filename: str, columns: List[str], data: List[str]):
    """Write a CSV file with the given filename, columns and data

    :param filename: The name of the file to write (without the extension)
    :type filename: str
    :param columns: The names of the columns
    :type columns: list[str]
    :param data: The data to write
    :type data: list[str]

    """
    with open(Path(OUTPUT_PATH).joinpath(filename + ".csv"), "w", encoding="utf8") as csv_f:
        csv_f.write(",".join(columns) + "\n")
        for row in data:
            csv_f.write(row)


def solve_and_write_csv(exercise: callable, columns: List[str], filename:str):
    data = exercise()
    write_CSV(filename, columns, data)


def ej2a(pokemons: List[Pokemon], config: ConfigData):

    data:List[str] = []

    for pokeball in config.pokeballs:
        for health in config.healths:
            poke_prob_avg, poke_stdev = stress_pokeball(pokeball, list(filter(lambda pokemon: math.ceil((pokemon.current_hp * 100)/pokemon.max_hp) == math.floor(health * 100), pokemons)), config.iterations)
            data.append(f"{pokeball},{health},{poke_prob_avg},{poke_stdev}\n")

    print("Writing CSV...")
    return data

# similar al ej2a pero en este caso tomamos 2 pokemons y vemos como varia con la health
def ej2b(pokemons: List[Pokemon], config: ConfigData):
    data: List[str] = []
    
    for pokeball in config.pokeballs:
        for pokemon_name in config.pokemon_names:
            for health in config.healths:
                poke_prob_avg, poke_stdev = stress_pokeball(pokeball, list(filter(lambda pokemon: pokemon.name == pokemon_name and math.ceil((pokemon.current_hp * 100)/pokemon.max_hp) == math.floor(health * 100), pokemons)), config.iterations)
                data.append(f"{pokeball},{pokemon_name},{health},{poke_prob_avg},{poke_stdev}\n")
    print("Writing CSV...")
    return data


def ej2c(pokemons: List[Pokemon], config: ConfigData):
    pass
def ej2d(pokemons: List[Pokemon], config: ConfigData):
    pass
def ej2e(pokemons: List[Pokemon], config: ConfigData):
    pass

if __name__ == "__main__":
    output_path = Path(OUTPUT_PATH)

    config = load_config()

    # pokemons = create_all_pokemons(
    #     config.pokemon_names,
    #     config.levels,
    #     config.status_effects,
    #     config.healths
    # )
    os.makedirs(output_path, exist_ok=True)  # create dir if not exists
    ej1b_data()
    ej1b_plot()
    #solve_and_write_csv(lambda: ej2a(pokemons, config), ["pokeball", "health", "avg_prob", "stdev"], "Ej2a")
    #solve_and_write_csv(lambda: ej2b(pokemons, config), ["pokeball", "pokemon_name", "health", "avg_prob", "stdev"], "Ej2b")
