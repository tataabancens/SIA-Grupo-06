import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objs as go
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
                 radius_update: RadiusUpdate = IdentityUpdate(),
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

    groups = [[] for _ in range(K**2)]

    for i in range(len(inputs)):
        winners[i] = kohonen.get_winner(inputs[i])  # [0, k**2)
        groups[int(winners[i])].append(countries[i])

    groups_dict = {f"Group {i}": g for i,g in enumerate(groups)}

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

    group_names = np.array([", ".join(groups[i]) if len(groups[i]) > 0 else "Empty group" for i in range(K**2)]).reshape((K, K))

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Groups", "UDM"))
    groups_heatmap = go.Heatmap(
                    z=np.array(list(map(lambda x: len(x), groups))).reshape((K, K)),
                    text=group_names,
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    colorscale='Viridis',
                    colorbar=dict(title='Scale 1', x=0.45))

    udm_heatmap = go.Heatmap(z=kohonen.get_unified_distance_matrix(),
                             colorscale='Greys',
                             colorbar=dict(title='Scale 2', x=1))

    fig.add_trace(groups_heatmap, row=1, col=1)
    fig.add_trace(udm_heatmap, row=1, col=2)
    fig.update_layout(
        title='Grouping of countries'
    )

    fig.show()


if __name__ == '__main__':
    main()
