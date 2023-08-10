import json

from typing import List
import sys
from pathlib import Path
import os
from statistics import mean, stdev

from src.catching import attempt_catch
from src.pokemon import PokemonFactory, StatusEffect, Pokemon
from utils.constants import OUTPUT_PATH, DEFAULT_NOISE, EJ1_FILENAME


class ConfigData:
    iterations: int = 100
    pokeballs: List[str] = ["pokeball", "ultraball", "fastball", "heavyball"]
    pokemon_names: List[str] = ["snorlax"]
    level: int = 100
    status_effect: str = "none"
    health: float = 1.0


def stress_pokeball(ball: str, pkmons: List[Pokemon], n: int):
    """Tests the given Pokeball with the provided list of Pokemons and returns the average catch rate and the standard
    deviation

    :param ball: The name of the type of ball to use
    :type ball: str
    :param pkmons: List of Pokemons to try catching
    :type pkmons: list[Pokemon]
    :param n: The amount of iterations to test
    :type n: int
    :return:
        The catch rate of the Pokeball in relation to the given Pokemons, as a tuple with the mean catch rate and
        its standard deviation
    """
    catch_rates = []
    catches = 0
    for pkmon in pkmons:
        for _ in range(n):
            caught, rate = attempt_catch(pkmon, ball, DEFAULT_NOISE)
            catch_rates.append(rate)
            catches += 1 if caught else 0
    print(f"For {pokeball}: {catches}/{n*len(pkmons)} catches")
    mean_rate = mean(catch_rates)
    return [mean_rate, stdev(catch_rates, mean_rate)]


def create_all_pokemons(names: List[str], lvl: int, status: StatusEffect, health: float) -> List[Pokemon]:
    factory = PokemonFactory("pokemon.json")
    pokemons_to_ret: List[Pokemon] = []
    for pokemon_name in names:
        pokemons_to_ret.append(factory.create(pokemon_name, lvl, status, health))
    return pokemons_to_ret


def load_config() -> ConfigData:
    config_data = ConfigData()
    if len(sys.argv) == 1:
        return config_data

    with open(f"{sys.argv[1]}", "r") as config_f:
        config = json.load(config_f)

        # With default values
        try:
            config_data.iterations = config["iterations"]
        except KeyError:
            pass
        try:
            config_data.pokeballs = config["pokeballs"]
        except KeyError:
            pass
        try:
            config_data.pokemon_names = config["pokemons"]
        except KeyError:
            pass
        try:
            config_data.level = config["level"]
        except KeyError:
            pass
        try:
            config_data.status_effect = config["status_effect"]
        except KeyError:
            pass
        try:
            config_data.health = config["health"]
        except KeyError:
            pass
    return config_data


if __name__ == "__main__":
    total_prob_sum, total_max_prob, total_min_prob = 0, 0, 1
    output_path = Path(OUTPUT_PATH)

    config = load_config()

    pokemons = create_all_pokemons(
        config.pokemon_names,
        config.level,
        StatusEffect.from_value(config.status_effect),
        config.health
    )

    os.makedirs(output_path, exist_ok=True)  # create dir if not exists
    with open(output_path.joinpath(EJ1_FILENAME), "w") as csv_f:

        csv_f.write("pokeball,avg_prob,stdev\n")

        # TODO: ahora está todo hardcodeado pero habría que ver si podemos abstraer todo en el config file
        #       o directamente hacer todas las rutinas hardcodeadas acá para cada gráfico

        for pokeball in config.pokeballs:
            poke_prob_avg, poke_stdev = stress_pokeball(pokeball, pokemons, config.iterations)
           # TODO: Imprimir esto en el csv en vez de en consola  || por qué no ambos? =D
            print(f"Pokebola: {pokeball}")
            print(f"average_prob: {poke_prob_avg}")
            print(f"deviation: {poke_stdev}")
            print("---------------------")
            csv_f.write(f"{pokeball},{poke_prob_avg},{poke_stdev}\n")
