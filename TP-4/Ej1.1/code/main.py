import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import seaborn as sns

import numpy as np
import plotly.graph_objs as go
from matplotlib import pyplot as plt

from plotly.subplots import make_subplots

from config.config import Input, load_config, ConfigPath, KohonenConfig
from radius_update import IdentityUpdate, ProgressiveReduction, RadiusUpdate
from similarity import EuclideanSimilarity, Similarity
from standardization import ZScore
from utils.io import get_src_str


class KohonenNetwork:

    def __init__(self,
                 k: int,
                 r: float,
                 input_size: int,
                 learning_rate: float,
                 max_epochs: int,
                 initial_input: Optional[list] = None,
                 radius_update: RadiusUpdate = ProgressiveReduction(),
                 similarity: Similarity = EuclideanSimilarity(),
                 ):
        self.input = np.zeros(input_size)
        self.input_size = input_size
        self.k = k
        self.radius = r
        self.original_radius = r
        self.radius_update = radius_update
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.similarity = similarity


        self.weights = self.initialize_weights(k, input_size, initial_input)

    @staticmethod
    def initialize_weights(k: int, input_size: int, weights: Optional[list]) -> np.ndarray:
        if weights is None:
            return np.random.uniform(0, 1, k**2 * input_size).reshape((k**2, input_size))
        return np.array(weights).reshape((k**2, input_size))

    def train(self, data: np.ndarray):
        for current_epoch in range(0, self.max_epochs):

            self.input = data[np.random.randint(0, len(data))]

            winner_neuron = self.get_winner(self.input)
            self.update_weights(winner_neuron)

            self.radius = self.radius_update.update(self.original_radius, current_epoch)

    def update_weights(self, neuron):
        neighbourhood = self.get_neighbourhood(neuron)
        for i in neighbourhood:
            self.weights[i] = self.weights[i] + self.learning_rate * (self.input - self.weights[i])

    def get_neighbourhood(self, neuron_index: int) -> list:
        """given the index of a neuron, returns an array of all the neighbouring neurons inside the current radius"""
        neighbours = []
        matrix_shape = (self.k, self.k)
        neuron_coords_array = np.array(np.unravel_index(neuron_index, matrix_shape))
        for i in range(0, self.k**2):
            coord = np.unravel_index(i, matrix_shape)
            distance = np.linalg.norm(neuron_coords_array - np.array(coord))
            if distance <= self.radius:
                neighbours.append(i)
        return neighbours

    def get_winner(self, input_data) -> int:
        weights_size = self.k**2
        distances = np.zeros(weights_size)
        for i in range(weights_size):
            distances[i] = self.similarity.calculate(input_data, self.weights[i])

        return int(np.argmin(distances))

    def __get_direct_neighbours_coords(self, index: int):
        matrix_shape = (self.k, self.k)
        coord = np.unravel_index(index, matrix_shape)
        neighbours = []
        possible_neighbours = [(coord[0], coord[1] + 1), (coord[0], coord[1] - 1), (coord[0] + 1, coord[1]), (coord[0] - 1, coord[1])]
        for p in possible_neighbours:
            if 0 <= p[0] <= self.k - 1 and 0 <= p[1] <= self.k - 1:
                neighbours.append(np.ravel_multi_index(p, matrix_shape))
        return neighbours

    def get_unified_distance_matrix(self):
        matrix = np.zeros((self.k, self.k))

        for i in range(self.k**2):
            neighbours = self.__get_direct_neighbours_coords(i)
            i_coord = np.unravel_index(i, (self.k, self.k))
            matrix[i_coord] = np.mean(list(map(lambda n: np.linalg.norm(self.weights[i] - self.weights[n]), neighbours)))
        return matrix


def main():
    inputs = Input()
    inputs.load_from_csv("europe.csv")
    inputs.clean_input()

    countries = [inputs.data[i][0] for i in range(len(inputs.data))]
    areas = [inputs.data[i][1] for i in range(len(inputs.data))]
    gdps = [inputs.data[i][2] for i in range(len(inputs.data))]
    inflations = [inputs.data[i][3] for i in range(len(inputs.data))]
    life_expectancies = [inputs.data[i][4] for i in range(len(inputs.data))]
    military_expenditures = [inputs.data[i][5] for i in range(len(inputs.data))]
    populations = [inputs.data[i][6] for i in range(len(inputs.data))]
    unemployments = [inputs.data[i][7] for i in range(len(inputs.data))]

    standarization = ZScore()

    inputs = np.array([standarization.standardize(inputs.clear_data[i]) for i in range(len(inputs.clear_data))])

    config: KohonenConfig = load_config(ConfigPath.EJ1_1, "template.json")

    K = config.grid_size
    R = config.neighbours_radius
    LEARNING_RATE = config.learning_rate
    INPUT_SIZE = inputs.shape[1]
    MAX_EPOCHS = config.epochs

    initial_weights = []
    for i in range(K**2):
        initial_weights.extend(inputs[np.random.randint(0, len(inputs))])

    kohonen = KohonenNetwork(
        K,
        R,
        INPUT_SIZE,
        LEARNING_RATE,
        MAX_EPOCHS,
        initial_input=initial_weights,
        radius_update=ProgressiveReduction(),
        similarity=EuclideanSimilarity())

    kohonen.train(inputs)

    winners = np.zeros(len(inputs))

    country_groups = [[] for _ in range(K**2)]
    areas_groups = [[] for _ in range(K**2)]
    gdp_groups = [[] for _ in range(K**2)]
    inflation_groups = [[] for _ in range(K**2)]
    life_expectancy_groups = [[] for _ in range(K**2)]
    military_expenditure_groups = [[] for _ in range(K**2)]
    population_groups = [[] for _ in range(K**2)]
    unemployment_groups = [[] for _ in range(K**2)]


    for i in range(len(inputs)):
        winners[i] = kohonen.get_winner(inputs[i])  # [0, k**2)
        country_groups[int(winners[i])].append(countries[i])
        areas_groups[int(winners[i])].append(int(areas[i]))
        gdp_groups[int(winners[i])].append(int(gdps[i]))
        inflation_groups[int(winners[i])].append(float(inflations[i]))
        life_expectancy_groups[int(winners[i])].append(float(life_expectancies[i]))
        military_expenditure_groups[int(winners[i])].append(float(military_expenditures[i]))
        population_groups[int(winners[i])].append(float(populations[i]))
        unemployment_groups[int(winners[i])].append(float(unemployments[i]))

    for i in range(len(areas_groups)):
        areas_groups[i] = np.mean(areas_groups[i]) if len(areas_groups[i]) > 0 else 0
        gdp_groups[i] = np.mean(gdp_groups[i]) if len(gdp_groups[i]) > 0 else 0
        inflation_groups[i] = np.mean(inflation_groups[i]) if len(inflation_groups[i]) > 0 else 0
        life_expectancy_groups[i] = np.mean(life_expectancy_groups[i]) if len(life_expectancy_groups[i]) > 0 else 0
        military_expenditure_groups[i] = np.mean(military_expenditure_groups[i]) if len(military_expenditure_groups[i]) > 0 else 0
        population_groups[i] = np.mean(population_groups[i]) if len(population_groups[i]) > 0 else 0
        unemployment_groups[i] = np.mean(unemployment_groups[i]) if len(unemployment_groups[i]) > 0 else 0


    groups_dict = {f"Group {i}": g for i,g in enumerate(country_groups)}


    output_directory = Path(get_src_str(), "Ej1.1", "output")
    os.makedirs(output_directory, exist_ok=True)

    with open(Path(get_src_str(), "Ej1.1", "output", f"result-{datetime.now()}.json"), "w", encoding="utf-8") as file:
        result = {
            "config": config.to_json(),
            "dataset": {
                "name": "europe.csv",
                "input_size": INPUT_SIZE,
                "size": len(inputs)
            },
            "groups": groups_dict,
            "weights": kohonen.weights.tolist()
        }
        json.dump(result, file, ensure_ascii=False, indent=4)

    ### --- HEATMAP CON NOMBRES -----
    matrix = np.zeros((K, K), dtype=int)

    # Process the data and fill the matrix
    for group, countries in groups_dict.items():
        if countries:
            row, col = divmod(int(group.split()[1]), K)
            matrix[row, col] = len(countries)

    # Add group names to each cell
    plt.figure(figsize=(10, 8))
    for i in range(K):
        for j in range(K):
            plt.text(j + 0.5, i + 0.5, '\n'.join(groups_dict.get(f"Group {i * K + j}", "")), ha='center', va='center',
                     fontsize=10)

    plt.title(f"Groups Heatmap {K}x{K} with η(0)={str(LEARNING_RATE)}, R={str(R)} and {MAX_EPOCHS} epochs")

    groups = np.array(list(map(lambda x: len(x), country_groups))).reshape((K, K))
    groups = np.flip(groups, axis=0)

    sns.heatmap(groups, cmap='viridis', annot=False)
    plt.show()

    ####
    ### NEIGHBOURS ###

    plt.title(f"Unified Distance Matrix Heatmap")

    unified_distance = kohonen.get_unified_distance_matrix()
    unified_distance = np.flip(unified_distance, axis=0)

    sns.heatmap(unified_distance, cmap='gray', annot=True)
    plt.show()

    ####





    group_names = np.array([", ".join(country_groups[i]) if len(country_groups[i]) > 0 else "Empty group" for i in range(K**2)]).reshape((K, K))
    group_areas = np.array([str(areas_groups[i]) for i in range(K**2)]).reshape((K, K))
    group_gdps = np.array([str(gdp_groups[i]) for i in range(K**2)]).reshape((K, K))
    group_inflations = np.array([str(inflation_groups[i]) for i in range(K**2)]).reshape((K, K))
    group_life_expectancies = np.array([str(life_expectancy_groups[i]) for i in range(K**2)]).reshape((K, K))
    group_military_expenditures = np.array([str(military_expenditure_groups[i]) for i in range(K**2)]).reshape((K, K))
    group_populations = np.array([str(population_groups[i]) for i in range(K**2)]).reshape((K, K))
    group_unemployments = np.array([str(unemployment_groups[i]) for i in range(K**2)]).reshape((K, K))

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Groups", "UDM"))
    groups_heatmap = go.Heatmap(
                    z=np.array(list(map(lambda x: len(x), country_groups))).reshape((K, K)),
                    text=group_names,
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    colorscale='Viridis',
                    colorbar=dict(x=0.45))

    udm_heatmap = go.Heatmap(z=kohonen.get_unified_distance_matrix(),
                             colorscale='Greys',
                             colorbar=dict(x=1))


    area_and_gdps_fig = make_subplots(rows=1, cols=2, subplot_titles=("Areas", "GDPS"))
    inflation_and_like_fig = make_subplots(rows=1, cols=2, subplot_titles=("Inflations", "Life expectancies"))
    military_and_pop_fig = make_subplots(rows=1, cols=2, subplot_titles=("Military expenditures", "Populations"))
    unemployment_fig = make_subplots(rows=1, cols=2, subplot_titles=("Unemployments"))

    countries_per_area_heatmap = go.Heatmap(
        z=np.array([areas_groups[i] for i in range(K**2)]).reshape((K, K)),
        text=group_areas,
        colorscale='Viridis',
        texttemplate="%{text:.2f}",
        textfont={"size": 20},
        colorbar=dict(x=0.45)
    )

    area_and_gdps_fig.add_trace(countries_per_area_heatmap, row=1, col=1)

    countries_per_gdp_heatmap = go.Heatmap(
        z=np.array([gdp_groups[i] for i in range(K**2)]).reshape((K, K)),
        text=group_gdps,
        colorscale='Viridis',
        texttemplate="%{text:.2f}",
        textfont={"size": 20},
        colorbar=dict(x=1))

    area_and_gdps_fig.add_trace(countries_per_gdp_heatmap, row=1, col=2)

    countries_per_inflation_heatmap = go.Heatmap(
        z=np.array([inflation_groups[i] for i in range(K**2)]).reshape((K, K)),
        text=group_inflations,
        colorscale='Viridis',
        texttemplate="%{text:.2f}",
        textfont={"size": 20},
        colorbar=dict(x=0.45))

    inflation_and_like_fig.add_trace(countries_per_inflation_heatmap, row=1, col=1)

    countries_per_life_expectancy_heatmap = go.Heatmap(
        z=np.array([life_expectancy_groups[i] for i in range(K**2)]).reshape((K, K)),
        text=group_life_expectancies,
        colorscale='Viridis',
        texttemplate="%{text:.2f}",
        textfont={"size": 20},
        colorbar=dict(x=1))

    inflation_and_like_fig.add_trace(countries_per_life_expectancy_heatmap, row=1, col=2)

    countries_per_military_expenditure_heatmap = go.Heatmap(
        z=np.array([military_expenditure_groups[i] for i in range(K**2)]).reshape((K, K)),
        text=group_military_expenditures,
        colorscale='Viridis',
        texttemplate="%{text:.2f}",
        textfont={"size": 20},
        colorbar=dict(x=0.45))

    military_and_pop_fig.add_trace(countries_per_military_expenditure_heatmap, row=1, col=1)

    countries_per_population_heatmap = go.Heatmap(
        z=np.array([population_groups[i] for i in range(K**2)]).reshape((K, K)),
        text=group_populations,
        colorscale='Viridis',
        texttemplate="%{text:.2f}",
        textfont={"size": 20},
        colorbar=dict(x=1))

    military_and_pop_fig.add_trace(countries_per_population_heatmap, row=1, col=2)

    countries_per_unemployment_heatmap = go.Heatmap(
        z=np.array([unemployment_groups[i] for i in range(K**2)]).reshape((K, K)),
        text=group_unemployments,
        colorscale='Viridis',
        texttemplate="%{text:.2f}",
        textfont={"size": 20},
        colorbar=dict(x=0.45))

    unemployment_fig.add_trace(countries_per_unemployment_heatmap, row=1, col=1)


    area_and_gdps_fig.update_layout(
        title='Areas and GDPS per group'
    )

    inflation_and_like_fig.update_layout(
        title='Inflation and life expectancies per group'
    )

    military_and_pop_fig.update_layout(

        title='Military expenditures and populations per group'
    )

    unemployment_fig.update_layout(
        title='Unemployment per group'
    )



    fig.add_trace(groups_heatmap, row=1, col=1)
    fig.add_trace(udm_heatmap, row=1, col=2)
    fig.update_layout(
        title='Grouping of countries'
    )

    fig.show()
    area_and_gdps_fig.show()
    inflation_and_like_fig.show()
    military_and_pop_fig.show()
    unemployment_fig.show()


if __name__ == '__main__':
    main()
