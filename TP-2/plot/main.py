import json
import os

import pandas as pd
from matplotlib import pyplot as plt
import sys
sys.path.append('../')
from config import load_config
from simulation import Simulation

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
            plot_lines(role, config_hash)


def plot_lines(role, hash):  # misma configs o sea mismo hash con dif fecha
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
        ys = [csv_object.at[i, 'fitness'] for i in range(csv_object.shape[0])]
        plt.errorbar(xs, ys, label=hash)

    plt.title(f"Desempeño por generación para {len(output_files)} iteraciones")
    plt.xlabel("generación")
    plt.ylabel("desempeño")
    plt.tight_layout()
    plt.annotate(f"Hash de configuración: {hash}", xy=(0.5, -0.15), xycoords='axes fraction', fontsize=10, color='gray')
    plt.show()

def run_simulations():
    config_paths = ['archer_config.json',
                    'fighter_config.json','infiltrate_config.json','defender_config.json']
    common_paths = ['standard_config_1.json', 'standard_config_2.json',
                    'standard_config_3.json','standard_config_4.json']

    for config_path in config_paths:
        path = os.getcwd() + '/config/' + config_path
        config = load_config(path)
        simulation = Simulation(n=config.N, crossover=config.crossover, selections=config.selections,
                                mutation=config.mutation, selection_strategy=config.selection_strategy,
                                crossover_proportion=config.A, selection_proportion=config.B, k=config.K, role=config.role,
                                max_iterations=config.max_iterations,
                                max_generations_without_improvement=config.max_iterations_without_change,
                                bolzmann_temperature=config.bolzmann_temperature,
                                deterministic_tournament_m=config.deterministic_tournament_m,
                                probabilistic_tournament_threshold=config.probabilistic_tournament_threshold,
                                plot=config.plot, plot_batch_size=config.plot_batch_size, config_path=path)
        simulation.run()
    for config_path in common_paths:
        for role in ['Archer','Infiltrate','Defender','Fighter']:
            path = os.getcwd() + '/config/' + config_path
            with open(path, 'r') as file:
                data = json.load(file)
            data['role'] = role
            with open(path, 'w') as file:
                json.dump(data, file, indent=4)
            config = load_config(path)
            simulation = Simulation(n=config.N, crossover=config.crossover, selections=config.selections,
                                    mutation=config.mutation, selection_strategy=config.selection_strategy,
                                    crossover_proportion=config.A, selection_proportion=config.B, k=config.K, role=config.role,
                                    max_iterations=config.max_iterations,
                                    max_generations_without_improvement=config.max_iterations_without_change,
                                    bolzmann_temperature=config.bolzmann_temperature,
                                    deterministic_tournament_m=config.deterministic_tournament_m,
                                    probabilistic_tournament_threshold=config.probabilistic_tournament_threshold,
                                    plot=config.plot, plot_batch_size=config.plot_batch_size, config_path=path)
            simulation.run()


def main():
    # plot_lines([out_path+"output_Fighter_453bca740c30a03ac81476df14dbac9a_2023-09-09-13-02-35.csv"])
    # print(output_files)
    run_simulations()
    plot_all_lines()


if __name__ == "__main__":
    main()
