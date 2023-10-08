import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pn


def plot_boxplot(data, box_plot_title, labels):
    data_arr = np.array(data)
    fig, ax = plt.subplots(figsize=(10, 7))
    x = np.array(labels)
    ax.set_xticklabels(x)
    plt.title(box_plot_title)
    ax.boxplot([data_arr[:, 0], data_arr[:, 1], data_arr[:, 2], data_arr[:, 3], data_arr[:, 4], data_arr[:, 5],
                data_arr[:, 6]],
               widths=0.5,
               boxprops=dict(color='black'),
               whiskerprops=dict(color='black'),
               medianprops=dict(color='red', linewidth=2))
    plt.show()

def plot_heatmap(inputs, countries, solver, k, learn_rate, radius, epochs):
    results = [solver.find_winner_neuron(i) for i in inputs]
    matrix = np.zeros((k, k))
    countries_matrix_aux = [["" for _ in range(k)] for _ in range(k)]

    for i in range(len(results)):
        matrix[results[i]] += 1
        countries_matrix_aux[results[i][0]][results[i][1]] += f"{countries[i]}\n"

    countries_matrix = np.array(countries_matrix_aux)

    plt.figure(figsize=(10, 8))
    plt.title(f"Heatmap {k}x{k} con η(0)={str(learn_rate)}, R(0)={str(radius)} y {epochs} epocas")
    sn.heatmap(matrix, cmap='Reds', annot=countries_matrix, fmt="")
    plt.show()
