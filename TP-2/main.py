"""
    Main file for the TP-2
"""
import statistics
import time

from config import load_config
from pathlib import Path
import argparse
from argparse import Namespace

from role import ItemStats, Stats, RoleType, Chromosome
from agent import Agent
from genetic import crossover
from simulation import Simulation


def __parse_args() -> Namespace:
    parser = argparse.ArgumentParser(
        prog='G06-TP2',
        description='Program to maximize a RPG with Genetic Algorithms'
    )
    parser.add_argument('-c', '--config',
                        type=str,
                        required=False,
                        nargs='?',
                        help='Path to the json configuration file',
                        dest='config',
                        action='store',
                        default=None)
    return parser.parse_args()


def main():
    args = __parse_args()
    config_path = Path(args.config if args.config is not None else './configs/configTemplate.json')
    if config_path is None:
        print("Config path not selected, using default")

    config = load_config(config_path)
    simulation = Simulation(n=config.N, crossover=config.crossover, selections=config.selections,
                            mutation=config.mutation, selection_strategy=config.selection_strategy,
                            crossover_proportion=config.A, selection_proportion=config.B, k=config.K, role=config.role,
                            max_iterations=config.max_iterations,
                            max_generations_without_improvement=config.max_iterations_without_change,
                            bolzmann_temperature=config.bolzmann_temperature,
                            deterministic_tournament_m=config.deterministic_tournament_m,
                            probabilistic_tournament_threshold=config.probabilistic_tournament_threshold,
                            plot=config.plot, plot_batch_size=config.plot_batch_size, config_path=config_path)
    simulation.run()


def main_deprecated():
    """
        Main function
    """
    args = __parse_args()
    config_path = Path(args.config if args.config is not None else './configs/configTemplate.json')
    if config_path is None:
        print("Config path not selected, using default")

    config = load_config(config_path)
    items = ItemStats(
        strength=config.items['strength'],
        agility=config.items['agility'],
        proficiency=config.items['proficiency'],
        toughness=config.items['toughness'],
        health=config.items['health'])
    items2 = ItemStats(
        strength=config.items['agility'],
        agility=config.items['strength'],
        proficiency=config.items['toughness'],
        toughness=config.items['health'],
        health=config.items['proficiency'])
    role = RoleType.get_instance_from_name(config.role_name)
    chromosome = Chromosome(items, 1.5)
    agent = Agent(role, chromosome)
    agent2 = Agent(role, Chromosome(items2, 1.9))
    my_tuple = crossover.OnePoint.cross((agent, agent2))
    tuple_as_string = ", ".join(str(item) for item in my_tuple)
    print(tuple_as_string)
    # Compute item stats from random weights
    computed_stats = ItemStats.from_weights(
        Stats(strength=45, agility=33.4, proficiency=12.3,
              toughness=1, health=9.6).get_as_list()
    )
    print(computed_stats)
    print(sum([computed_stats.strength, computed_stats.agility,
          computed_stats.proficiency, computed_stats.toughness, computed_stats.health]))
    print([1, 2])
    for i in range(1, 5, 2):
        print(i)


if __name__ == "__main__":
    main()
