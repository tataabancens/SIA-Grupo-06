"""
    Main file for the TP-2
"""
from config import load_config
from pathlib import Path
import argparse
from argparse import Namespace
from role import ItemStats, Stats, RoleType
from agent import Agent


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
    config_path = Path(args.config)
    if config_path is None:
        print("Config path not selected, using default")

    config = load_config(config_path)
    print(config)
    items = ItemStats(
        strength=config.items['strength'],
        agility=config.items['agility'],
        proficiency=config.items['proficiency'],
        toughness=config.items['toughness'],
        health=config.items['health'])
    role = RoleType.get_instance_from_name(config.role_name)

    agent = Agent(role, items)
    print(agent.compute_performance(1.5))

    # Compute item stats from random weights
    computed_stats = ItemStats.from_weights(
        Stats(strength=45, agility=33.4, proficiency=12.3, toughness=1, health=9.6)
    )
    print(computed_stats)
    print(sum([computed_stats.strength, computed_stats.agility, computed_stats.proficiency, computed_stats.toughness, computed_stats.health]))


if __name__ == "__main__":
    main()
