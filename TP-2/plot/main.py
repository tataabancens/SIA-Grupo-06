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


def plot_bars(out_path, role, y_value, y_label, x_label, hash_to_id):
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
        labels.append(hash_to_id.get(config_hash,f"{config_hash[:3]}"))

        values.append(np.mean(vals))
        std_devs.append(np.std(vals))

    # Create the bar chart with error bars
    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, alpha=0.5,
            ecolor="black", yerr=std_devs, capsize=5)
    plt.xticks(rotation=20, fontsize=10)
    plt.ylabel(y_label)
    plt.xlabel(x_label.capitalize())
    plt.title(f"{y_label.capitalize()} / {x_label} para {role} ({iters} iteraciones)")
    plt.subplots_adjust(bottom=0.3)

    plt.show()


def plot_all_bars(out_path, x_label, hash_to_id=None):
    if hash_to_id is None:
        hash_to_id = {}
    roles = set()
    for filename in os.listdir(out_path):
        parts = filename.split('_')
        roles.add(parts[1])
    for role in roles:
        plot_bars(out_path, role, 'fitness', 'desempeño', x_label, hash_to_id)
    for role in roles:
        plot_bars(out_path, role, 'diversity', 'diversidad', x_label, hash_to_id)


def plot_all_lines(out_path,  hash_to_id=None):
    if hash_to_id is None:
        hash_to_id = {}
    roles = set()
    config_hashes = set()
    for filename in os.listdir(out_path):
        parts = filename.split('_')
        roles.add(parts[1])
        config_hashes.add(parts[2])
    for role in roles:
        for config_hash in config_hashes:
            plot_lines(out_path, role, config_hash, 'diversity', 'diversidad', hash_to_id)
            plot_lines(out_path, role, config_hash, 'fitness', 'desempeño', hash_to_id)


def plot_lines(out_path, role, hash, y_value, y_label,  hash_to_id):  # misma configs o sea mismo hash con dif fecha
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

    plt.title(f"{y_label.capitalize()} / generación para {role} ({len(output_files)} iteraciones)")
    plt.xlabel("generación")
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.annotate(f"Configuración: {hash_to_id.get(hash,hash[:3])}", xy=(0.6, -0.15), xycoords='axes fraction', fontsize=10,
                 color='gray')
    plt.show()


def run_simulations():
    config_paths = ['infiltrate_config.json']
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
                                plot=config.plot, plot_batch_size=config.plot_batch_size, config_path=path,
                                pm=config.pm)
        simulation.run()
        print(f"finished {config_path}")
    for config_path in common_paths:
        for role in ['Infiltrate']:  # , 'Infiltrate', 'Defender', 'Fighter'
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
                                    plot=config.plot, plot_batch_size=config.plot_batch_size, config_path=path,
                                    pm=config.pm)
            simulation.run()
            print(f"finished {config_path}")


def main():
    # hash_to_id =  {
    #     "4f1fce139a92df545e983c172434654b": "OneGen",
    #     "9630e8f07e0c2a30178eede1f1f089c7": "LimitedMultiGen",
    #     "13818367a92e77616a785571dee792e3": "Complete",
    #     "a4cc5591326b7195f3e8d291e98d0d8b": "UniformMultiGen",
    # }
    # x_label = "método de mutación"
    # out_path = os.getcwd() + "/out_mutation_method_change/"
    #
    #
    # hash_to_id =  {
    #     "219c669ff7ea88fdef2f192aeac2470c": "0.1",
    #     "624657b563f8bda4bf78a96c9cf23c56": "0.3",
    #     "369310487c7fb606897d4521a1d4bc94": "0.9",
    #     "af1e83991dd73f52b6f9a5338f37eadd": "0.5",
    #     "f130edac4c72124a69d19906b10b27ba": "0.6"
    # }
    # x_label = "probabilidad de mutación"
    # out_path = os.getcwd() + "/out_mutation_change/"


    hash_to_id =  {
        "032a5af6ec39d3262c88837e3719e027": "config1",
        "cfa7f58afa56cda28cfde484ad6c0e6d": "óptima",
        "de2825ea4845e82a0025afbf7933c19c": "config4",
        "e118d690c26241708b5e760069214796": "config2",
        "f2197fa4fdff6e42277c2c725622b1e9": "config3"
    }
    x_label = "configuración"
    out_path = os.getcwd() + "/out/"

    # for i in range(15):
    #     run_simulations()

    plot_all_lines(out_path,  hash_to_id)

    plot_all_bars(out_path, x_label, hash_to_id)


if __name__ == "__main__":
    main()
