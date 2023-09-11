import json
import os

import pandas as pd
from matplotlib import pyplot as plt
import sys
import numpy as np

from typing import Dict, List

sys.path.append('../')
from config import load_config
from simulation import Simulation


def plot_bars(role, y_value, y_label):
    out_path = os.getcwd() + "/out/"
    identifier = "output_" + role
    fitness_by_hash: Dict[str, List] = {}
    for filename in os.listdir(out_path):
        file_path = os.path.join(out_path, filename)
        if os.path.isfile(file_path) and identifier in filename:
            hash_value = filename.split('_')[2]
            csv = pd.read_csv(file_path)
            fitness = csv.at[csv.shape[0] - 1, y_value]
            fitness_by_hash.setdefault(hash_value, [])
            fitness_by_hash[hash_value].append(fitness)

    labels = []
    values = []
    std_devs = []
    iters = 0
    for config_hash, vals in fitness_by_hash.items():
        if iters == 0:
            iters = len(vals)
        labels.append(f"{config_hash[:3]}")

        values.append(np.mean(vals))
        std_devs.append(np.std(vals))

    # Create the bar chart with error bars
    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, alpha=0.5,
            ecolor="black", yerr=std_devs, capsize=5)
    plt.xticks(rotation=20, fontsize=10)
    plt.ylabel(y_label)
    plt.xlabel(f"Configuración")
    plt.title(f"{y_label.capitalize()} por configuración para {role} ({iters} iteraciones)")
    plt.subplots_adjust(bottom=0.3)

    plt.show()


def plot_all_bars():
    out_path = os.getcwd() + "/out/"
    roles = set()
    for filename in os.listdir(out_path):
        parts = filename.split('_')
        roles.add(parts[1])
    for role in roles:
        plot_bars(role, 'fitness','desempeño')
    for role in roles:
        plot_bars(role, 'diversity','diversidad')


def plot_all_lines():
    out_path = os.getcwd() + "/out/"
    print(out_path)
    roles = set()
    config_hashes = set()
    for filename in os.listdir(out_path):
        parts = filename.split('_')
        roles.add(parts[1])
        config_hashes.add(parts[2])
    for role in roles:
        for config_hash in config_hashes:
            plot_lines(role, config_hash, 'diversity', 'diversidad')
            plot_lines(role, config_hash, 'fitness', 'desempeño')


def plot_lines(role, hash, y_value, y_label):  # misma configs o sea mismo hash con dif fecha
    out_path = os.getcwd() + "/out/"
    output_files = []
    identifier = "output_" + role + "_" + hash
    for filename in os.listdir(out_path):
        file_path = os.path.join(out_path, filename)
        if os.path.isfile(file_path) and identifier in filename:
            output_files.append(file_path)

    # [csv, csv]
    for csv in output_files:
        csv_object = pd.read_csv(csv)
        xs = [csv_object.at[i, 'generation'] for i in range(csv_object.shape[0])]
        print(xs)
        ys = [csv_object.at[i, y_value] for i in range(csv_object.shape[0])]
        plt.errorbar(xs, ys, label=hash)

    plt.title(f"{y_label.capitalize()} por generación para {len(output_files)} iteraciones")
    plt.xlabel("generación")
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.annotate(f"Hash de configuración: {hash[:3]}", xy=(0.5, -0.15), xycoords='axes fraction', fontsize=10, color='gray')
    plt.show()


def run_simulations():
    config_paths = ['archer_config.json']
                    # 'fighter_config.json', 'infiltrate_config.json', 'defender_config.json']
    common_paths = ['standard_config_1.json', 'standard_config_2.json',
                    'standard_config_3.json', 'standard_config_4.json']

    for config_path in config_paths:
        path = os.getcwd() + '/configs/' + config_path
        config = load_config(path)
        simulation = Simulation(n=config.N, crossover=config.crossover, selections=config.selections,
                                mutation=config.mutation, selection_strategy=config.selection_strategy,
                                crossover_proportion=config.A, selection_proportion=config.B, k=config.K,
                                role=config.role,
                                max_iterations=config.max_iterations,
                                max_generations_without_improvement=config.max_iterations_without_change,
                                bolzmann_temperature=config.bolzmann_temperature,
                                deterministic_tournament_m=config.deterministic_tournament_m,
                                probabilistic_tournament_threshold=config.probabilistic_tournament_threshold,
                                plot=config.plot, plot_batch_size=config.plot_batch_size, config_path=path, pm=config.pm)
        simulation.run()
        print(f"finished {config_path}")
    for config_path in common_paths:
        for role in ['Archer']: #, 'Infiltrate', 'Defender', 'Fighter'
            path = os.getcwd() + '/configs/' + config_path
            with open(path, 'r') as file:
                data = json.load(file)
            data['role'] = role
            with open(path, 'w') as file:
                json.dump(data, file, indent=4)
            config = load_config(path)
            simulation = Simulation(n=config.N, crossover=config.crossover, selections=config.selections,
                                    mutation=config.mutation, selection_strategy=config.selection_strategy,
                                    crossover_proportion=config.A, selection_proportion=config.B, k=config.K,
                                    role=config.role,
                                    max_iterations=config.max_iterations,
                                    max_generations_without_improvement=config.max_iterations_without_change,
                                    bolzmann_temperature=config.bolzmann_temperature,
                                    deterministic_tournament_m=config.deterministic_tournament_m,
                                    probabilistic_tournament_threshold=config.probabilistic_tournament_threshold,
                                    plot=config.plot, plot_batch_size=config.plot_batch_size, config_path=path,pm=config.pm)
            simulation.run()
            print(f"finished {config_path}")


def main():
    # for i in range(15):
    #     run_simulations()
    plot_all_lines()
    plot_all_bars()


if __name__ == "__main__":
    main()
