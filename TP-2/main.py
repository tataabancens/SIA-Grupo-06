"""
    Main file for the TP-2
"""
import statistics
import time
import json

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
    parser.add_argument('-c', '--configs',
                        type=str,
                        required=False,
                        nargs='?',
                        help='Path to the json configuration file',
                        dest='configs',
                        action='store',
                        default=None)
    return parser.parse_args()


def main():
    args = __parse_args()
    config_path = Path(args.configs if args.configs is not None else './configs/configTemplate.json')
    if config_path is None:
        print("Config path not selected, using default")

    config = load_config(config_path)
    simulation = Simulation(n=config.N, crossover=config.crossover, selections=config.selections,
                            mutation=config.mutation, selection_strategy=config.selection_strategy,
                            crossover_proportion=config.A, selection_proportion=config.B, k=config.K, role=config.role,
                            max_iterations=config.max_iterations,
                            max_generations_without_improvement=config.max_iterations_without_change,
                            bolzmann_temperature=config.bolzmann_temperature,
                            pm=config.pm,
                            deterministic_tournament_m=config.deterministic_tournament_m,
                            probabilistic_tournament_threshold=config.probabilistic_tournament_threshold,
                            plot=config.plot, plot_batch_size=config.plot_batch_size, config_path=config_path)
    agent = simulation.run()


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    tiempo_transcurrido = end - start

    print(f"El programa tardó {tiempo_transcurrido} segundos en ejecutarse.")
