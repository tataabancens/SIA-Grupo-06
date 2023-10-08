
import math
import numpy as np
from typing import Optional, Callable
from abc import ABC, abstractmethod
from config.config import Input, load_config, ConfigData, ConfigPath
from icecream.icecream import ic
import pandas as pd
import plotly.graph_objs as go


class Similarity(ABC):
    @abstractmethod
    def calculate(self, input: np.ndarray, weights: np.ndarray) -> float:
        pass


class EuclideanSimilarity(Similarity):
    def calculate(self, input: np.ndarray, weights: np.ndarray) -> float:
        return np.linalg.norm(input - weights)


class ExponentialSimilarity(Similarity):
    def calculate(self, input: np.ndarray, weights: np.ndarray) -> float:
        return np.exp(math.pow(-np.linalg.norm(input - weights),2))


class Standardization(ABC):
    @abstractmethod
    def standardize(self, values: np.ndarray) -> np.ndarray:
        pass


class UnitLengthScaling(Standardization):
    def standardize(self, values: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(values)
        return np.divide(values, norm)


class MinMaxFeatureScaling(Standardization):
    def __init__(self, range = (0.0, 1.0)):
        self.range = range

    def standardize(self, values: np.ndarray) -> np.ndarray:
        min = np.ones(values.shape) * np.min(values)
        max = np.ones(values.shape) * np.max(values)
        return ((values - min) / (max-min)) * (self.range[1] - self.range[0]) + self.range[0]


class RadiusUpdate(ABC):
    @abstractmethod
    def update(self, original_radius: float, iteration: int):
        pass


class IdentityUpdate(RadiusUpdate):

    def update(self, original_radius: float, iteration: int):
        return original_radius


class ProgressiveReduction(RadiusUpdate):

    def update(self, original_radius: float, iteration: int):
        if iteration == 0:
            return original_radius
        return max(1 / iteration, 1)


class KohonenNetwork:

    def __init__(self,
                 k: int,
                 r: float,
                 input_size: int,
                 learning_rate: float,
                 max_epochs: int,
                 initial_input: Optional[list] = None,
                 radius_update: RadiusUpdate = IdentityUpdate(),
                 standardization: Standardization = UnitLengthScaling(),
                 similarity: Similarity = EuclideanSimilarity(),
                 ):
        self.input = np.zeros(input_size)
        self.input_size = input_size
        self.k = k
        self.radius = r
        self.original_radius = r
        self.weights = self.initialize_weights(k, input_size, initial_input)
        self.radius_update = radius_update
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.similarity = similarity
        self.standarization = standardization

    @staticmethod
    def initialize_weights(k: int, input_size: int, weights: Optional[list]) -> np.ndarray:
        if weights is None:
            return np.random.uniform(0, 1, k**2 * input_size).reshape((k**2, input_size))
        return np.array(weights).reshape((k**2, input_size))


    def train(self, data: np.ndarray):
        for current_epoch in range(0, self.max_epochs):

            self.input = data[np.random.randint(0, len(data))]
            # Estandarizamos por que sino no tiene sentido la comparacion
            self.input = self.standarization.standardize(self.input)

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
        max_limit = self.radius * self.k
        matrix_shape = (self.k, self.k)
        neuron_coords_array = np.array(np.unravel_index(neuron_index, matrix_shape))
        for i in range(int(neuron_index - max_limit), int(neuron_index + max_limit)):
            if i >= self.k**2 or i < 0:
                continue
            coord = np.unravel_index(i, matrix_shape)
            distance = np.linalg.norm(neuron_coords_array - np.array(coord))
            if distance < self.radius:
                neighbours.append(i)
        return neighbours

    def get_winner(self, input_data) -> int:
        weights_size = self.k**2
        distances = np.zeros(weights_size)
        for i in range(weights_size):
            distances[i] = self.similarity.calculate(input_data, self.weights[i])

        return int(np.argmin(distances))

    @staticmethod
    def test() -> None:
        network = KohonenNetwork(5, 1, 0, 0, 100)
        neighbourhood = network.get_neighbourhood(np.ravel_multi_index((5,5),(5,5)))
        print("Neighbourhood is:", neighbourhood)
        true_neighbourhood = [(6,5), (5,6), (4,5), (5,4)]
        assert neighbourhood == list(map(lambda c: np.ravel_multi_index(c, (5, 5)), true_neighbourhood))


def main():
    inputs = Input()
    inputs.load_from_csv("europe.csv")
    countries = [inputs.data[i][0] for i in range(len(inputs.data))]
    inputs.clean_input()

    config = load_config(ConfigPath.EJ1_1, "template.json")

    K = int(config.grid_size)
    R = config.neighbours_radius
    LEARNING_RATE = config.learning_rate
    INPUT_SIZE = inputs.data.shape[1]
    MAX_EPOCHS = config.epochs

    initial_weights = []
    for i in range(K**2):
        initial_weights.extend(inputs.data[np.random.randint(0, len(inputs.data))])

    kohonen = KohonenNetwork(K, R, INPUT_SIZE, LEARNING_RATE, MAX_EPOCHS, initial_input=initial_weights, standardization=UnitLengthScaling(), radius_update=ProgressiveReduction(),similarity=EuclideanSimilarity())
    kohonen.train(inputs.data)

    winners = np.zeros(len(inputs.data))

    groups = [[] for i in range(K**2)]

    for i in range(len(inputs.data)):
        winners[i] = kohonen.get_winner(inputs.data[i])  # [0, k**2)
        groups[int(winners[i])].append(countries[i])

    groups_dict = {f"Group {i}": g for i,g in enumerate(groups)}


    output_file = open("./Ej1.1/output/output.txt", "w")

    for w in kohonen.weights:
        output_file.write(" ".join(map(str, w)) + "\n")

    group_names = np.array([" ".join(groups[i]) if len(groups[i]) > 0 else "Empty group" for i in range(K**2)]).reshape((K, K))

    fig = go.Figure(data=go.Heatmap(
                    z=np.array(list(map(lambda x: len(x), groups))).reshape((K, K)),
                    text=group_names,
                    texttemplate="%{text}",
                    textfont={"size": 10}))

    fig.show()


if __name__ == '__main__':
    main()