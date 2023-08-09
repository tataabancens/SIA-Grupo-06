import json

from src.catching import attempt_catch
from src.pokemon import PokemonFactory, StatusEffect, Pokemon
import sys


def stress_pokeball(ball: str, pkmons: list[Pokemon], n: int):
    prob_sum, max_prob, min_prob = 0, 0, 1
    for pkmon in pkmons:
        for _ in range(n):
            catch_ok, prob = attempt_catch(pkmon, ball, 0.15)
            max_prob = max(max_prob, prob)
            min_prob = min(min_prob, prob)
            prob_sum += prob
    prob_avg = prob_sum / (n * len(pkmons))
    return [prob_avg, max_prob, min_prob]


def create_all_pokemons(names: list[str], lvl: int, status: StatusEffect, health: float) -> list[Pokemon]:
    pokemons_to_ret: list[Pokemon] = []
    for pokemon_name in names:
        pokemons_to_ret.append(factory.create(pokemon_name, lvl, status, health))
    return pokemons_to_ret


if __name__ == "__main__":
    factory = PokemonFactory("pokemon.json")
    snorlax = factory.create("snorlax", 100, StatusEffect.NONE, 1)
    total_prob_sum, total_max_prob, total_min_prob = 0, 0, 1

    with open(f"{sys.argv[1]}", "r") as config_f:
        config = json.load(config_f)

        iterations: int = config["iterations"]
        pokeballs: list[str] = config["pokeballs"]
        pokemon_names: list[str] = config["pokemons"]
        level: int = config["level"]
        status_effect: str = config["status_effect"]
        health: float = config["health"]

        pokemons = create_all_pokemons(pokemon_names, level, StatusEffect.from_value(status_effect), health)
        with open("output/Ej1.csv", "w") as csv_f:
            # TODO: Imprimir aca los headers del csv
            for pokeball in pokeballs:
                poke_prob_avg, poke_max_prob, poke_min_prob = stress_pokeball(pokeball, pokemons, iterations)
                total_prob_sum += poke_prob_avg
                total_max_prob = max(total_max_prob, poke_max_prob)
                total_min_prob = min(total_min_prob, poke_min_prob)
            # TODO: Imprimir esto en el csv en vez de en consola
                print(f"Pokebola: {pokeball}")
                print(f"min: {poke_min_prob}, max: {poke_max_prob}")
                print(f"average_prob: {total_prob_sum / len(pokeballs)}")
                print("---------------------")
