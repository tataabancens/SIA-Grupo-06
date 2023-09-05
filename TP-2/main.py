"""
    Main file for the TP-2
"""
from config import load_config
from pathlib import Path
import argparse
from argparse import Namespace
from role import ItemStats, Stats, RoleType, Cromosome
from agent import Agent
from genetic import crossover


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
    cromosome = Cromosome(items, 1.5)
    agent = Agent(role, cromosome)
    agent2 = Agent(role, Cromosome(items2,1.9))
    my_tuple = crossover.OnePoint.cross((agent, agent2))
    tuple_as_string = ", ".join(str(item) for item in my_tuple)
    print(tuple_as_string)
    # Compute item stats from random weights
    computed_stats = ItemStats.from_weights(
        Stats(strength=45, agility=33.4, proficiency=12.3, toughness=1, health=9.6).get_as_list()
    )
    print(computed_stats)
    print(sum([computed_stats.strength, computed_stats.agility, computed_stats.proficiency, computed_stats.toughness, computed_stats.health]))
    print([1,2])
    for i in range(1, 5, 2):
        print(i)

if __name__ == "__main__":
    main()
