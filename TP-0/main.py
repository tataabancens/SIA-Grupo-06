import json

from typing import List

from src.catching import attempt_catch
from src.pokemon import PokemonFactory, StatusEffect, Pokemon
import sys
import plotly.express as px
from pathlib import Path
import os

_NOISE = 0.15  # TODO: qué valor deberíamos usar??
_OUTPUT_HTML_NAME = "first_figure"


class ConfigData:
    iterations: int = 100
    pokeballs: List[str] = ["pokeball", "ultraball", "fastball", "heavyball"]
    pokemon_names: List[str] = ["snorlax"]
    level: int = 100
    status_effect: str = "none"
    health: float = 1.0


def plot_values(x: List[any], y: List[float], title="Title"):
    fig = px.bar(x=x, y=y, title=title)
    fig.update_layout(title_font_size=50)
    fig.write_html(f'{_OUTPUT_HTML_NAME}.html', auto_open=True)


def stress_pokeball(ball: str, pkmons: List[Pokemon], n: int):
    prob_sum, max_prob, min_prob = 0, 0, 1
    for pkmon in pkmons:
        for _ in range(n):
            catch_ok, prob = attempt_catch(pkmon, ball, _NOISE)
            max_prob = max(max_prob, prob)
            min_prob = min(min_prob, prob)
            prob_sum += prob
    prob_avg = prob_sum / (n * len(pkmons))
    return [prob_avg, max_prob, min_prob]


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
        config_data.iterations = config["iterations"]
        config_data.pokeballs = config["pokeballs"]
        config_data.pokemon_names = config["pokemons"]
        config_data.level = config["level"]
        config_data.status_effect = config["status_effect"]
        config_data.health = config["health"]
    return config_data


if __name__ == "__main__":
    total_prob_sum, total_max_prob, total_min_prob = 0, 0, 1
    output_path = Path("output")
    ej1_filename = "Ej1"

    config = load_config()

    pokemons = create_all_pokemons(
        config.pokemon_names,
        config.level,
        StatusEffect.from_value(config.status_effect),
        config.health
    )

    os.makedirs(output_path, exist_ok=True)  # create dir if not exists
    with open(output_path.joinpath(f"{ej1_filename}.csv"), "w") as csv_f:

        csv_f.write("pokeball,min_prob,max_prob,avg_prob")

        # TODO: ahora está todo hardcodeado pero habría que ver si podemos abstraer todo en el config file
        #       o directamente hacer todas las rutinas hardcodeadas acá para cada gráfico

        probs = []
        balls = []
        for pokeball in config.pokeballs:
            poke_prob_avg, poke_max_prob, poke_min_prob = stress_pokeball(pokeball, pokemons, config.iterations)
           # TODO: Imprimir esto en el csv en vez de en consola  || por qué no ambos? =D
            print(f"Pokebola: {pokeball}")
            print(f"min: {poke_min_prob}, max: {poke_max_prob}")
            print(f"average_prob: {poke_prob_avg}")
            print("---------------------")
            csv_f.write(f"{pokeball},{poke_min_prob},{poke_max_prob},{avg_prob}\n")

            balls.append(pokeball)
            probs.append(avg_prob)
        plot_values(balls, probs, "Balls")