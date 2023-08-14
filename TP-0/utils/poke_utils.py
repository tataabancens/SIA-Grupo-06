from statistics import mean, stdev

from src.catching import attempt_catch
from src.pokemon import PokemonFactory, StatusEffect, Pokemon
from utils.constants import OUTPUT_PATH, DEFAULT_NOISE
from typing import List, Tuple
from math import sqrt


def catch_pokemon(pokemon: Pokemon, pokeball_type: str, n: int) -> Tuple[float, float]:
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


def stress_pokeball_2(ball: str, pkmons: List[Pokemon], n: int) -> Tuple[float, float]:
    """Tests the given Pokeball with the provided list of Pokemons and returns the average catch rate and the standard error

    :param ball: The name of the type of ball to use
    :type ball: str
    :param pkmons: List of Pokemons to try catching
    :type pkmons: list[Pokemon]
    :param n: The amount of iterations to test
    :type n: int
    :return: (Tuple[float,float]) The catch rate of the Pokeball in relation to the given Pokemons, as a tuple with the mean catch rate and
        its standard error
    """
    catch_rates = []
    for pkmon in pkmons:
        catch_rates.append(catch_pokemon(pkmon, ball, n)[0])
    mean_rate = mean(catch_rates)
    return mean_rate, stdev(catch_rates, mean_rate) / sqrt(len(catch_rates))


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
    # print(f"For {pokeball}: {catches}/{n*len(pkmons)} catches")

    mean_rate = mean(catch_rates) if len(catch_rates) > 0 else 0
    variance = stdev(catch_rates, mean_rate) if len(catch_rates) > 0 else 0

    return [mean_rate, variance]


def create_all_pokemons_base(names: List[str], lvl: int, status: StatusEffect, health: float) -> List[Pokemon]:
    factory = PokemonFactory("../pokemon.json")
    pokemons_to_ret: List[Pokemon] = []
    for pokemon_name in names:
        pokemons_to_ret.append(factory.create(pokemon_name, lvl, status, health))
    return pokemons_to_ret


def create_all_pokemons(names: List[str], lvls: List[int], statuses: List[StatusEffect], healths: List[float]) -> List[
    Pokemon]:
    factory = PokemonFactory("../pokemon.json")
    pokemons_to_ret: List[Pokemon] = []
    for pokemon_name in names:
        for lvl in lvls:
            for status in statuses:
                for health in healths:
                    pokemons_to_ret.append(factory.create(pokemon_name, lvl, status, health))
    return pokemons_to_ret
