import os

import pandas as pd
from matplotlib import pyplot as plt


def plot_all_lines():
    out_path = os.getcwd() + "/../out/"
    roles = set()
    config_hashes = set()
    for filename in os.listdir(out_path):
        parts = filename.split('_')
        roles.add(parts[1])
        config_hashes.add(parts[2])
    for role in roles:
        for config_hash in config_hashes:
            plot_lines(role, config_hash)


def plot_lines(role, hash):  # misma config o sea mismo hash con dif fecha
    out_path = os.getcwd() + "/../out/"
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
    plt.show()


def main():
    role = "Fighter"
    hash = "453bca740c30a03ac81476df14dbac9a"

    # plot_lines([out_path+"output_Fighter_453bca740c30a03ac81476df14dbac9a_2023-09-09-13-02-35.csv"])
    # print(output_files)
    plot_all_lines()


if __name__ == "__main__":
    main()
