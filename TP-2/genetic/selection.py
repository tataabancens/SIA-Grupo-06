from enum import Enum
from abc import ABC, abstractmethod
from typing import Optional
from agent import Agent
import numpy as np


class Selection(ABC):
    """
        Selection method abstract class
    """

    @abstractmethod
    def select(self, population: list[Agent], K: int) -> list[Agent]:
        """
            Selects the individuals from the population
        """
        pass


class Elite(Selection):
    """
        Elite selection method
    """

    def select(self, population: list[Agent], K: int) -> list[Agent]:
        """
            Selects the individuals from the population
        """
        population_lenght = len(population)

        if K <= population_lenght:
            return population[0:K]

        population_with_repetition = []
        for i in range(K):
            population_with_repetition.append(
                population[i % population_lenght])


class Roulette(Selection):
    """
        Roulette selection method
    """

    def select(self, population: list[Agent], K: int) -> list[Agent]:
        """
            Selects the individuals from the population
        """
        random_numbers = np.random.uniform(0, 1, K)
        return Roulette().roulette_with_given_random(population, random_numbers)

    def roulette_with_given_random(self, population: list[Agent], random_numbers: list[float]) -> list[Agent]:
        """
            Selects the individuals from the population
        """
        fitness_per_agent = np.array(
            [agent.compute_performance() for agent in population])
        fitness_sum = np.sum(fitness_per_agent)

        relative_fitness = np.array(
            [fitness / fitness_sum for fitness in fitness_per_agent])
        cumulative_fitness = np.cumsum(relative_fitness)

        selected_agents = []
        for random_number in random_numbers:
            for i in range(len(cumulative_fitness)):
                if cumulative_fitness[i-1] < random_number <= cumulative_fitness[i]:
                    selected_agents.append(population[i])
                    break

        return selected_agents

    def roulette_with_give_fitness(self, population: list[Agent], K: int, fitness_per_agent: list[float]) -> list[Agent]:
        """
            Selects the individuals from the population
        """
        fitness_sum = np.sum(fitness_per_agent)

        relative_fitness = np.array(
            [fitness / fitness_sum for fitness in fitness_per_agent])
        cumulative_fitness = np.cumsum(relative_fitness)

        random_numbers = np.random.uniform(0, 1, K)

        selected_agents = []
        for random_number in random_numbers:
            for i in range(len(cumulative_fitness)):
                if cumulative_fitness[i-1] < random_number <= cumulative_fitness[i]:
                    selected_agents.append(population[i])
                    break

        return selected_agents


class Universal(Selection):
    """
        Universal selection method
    """

    def select(self, population: list[Agent], K: int) -> list[Agent]:
        """
            Selects the individuals from the population
        """
        r = np.random.uniform(0, 1)
        random_numbers = np.array(
            [(r + j) / K for j in range(K)])
        return Roulette().roulette_with_given_random(population, random_numbers)


class Boltzmann(Selection):
    """
        Boltzmann selection method
    """

    def select(self, population: list[Agent], K: int, **kargs) -> list[Agent]:
        """
            Selects the individuals from the population
        """
        if kargs.get["T"] is None:
            raise Exception("T is required")

        T = kargs.get["T"]

        relative_fitness = np.array(
            np.exp([agent.compute_performance() / T for agent in population]))

        fitness_avr = np.average(relative_fitness)

        fitness_per_agent = np.array(
            fitness/fitness_avr for fitness in relative_fitness)

        return Roulette().roulette_with_give_fitness(population, K, fitness_per_agent)


class ProbabilisticTournament(Selection):
    """
        Tournament selection method
    """

    def select(self, population: list[Agent], K: int, **kargs) -> list[Agent]:
        """
            Selects the individuals from the population
        """

        if kargs.get["Threshold"] is None:
            raise Exception("Threshold is required")

        threshold = kargs.get["Threshold"]

        selected_agents = []

        for i in range(K):
            random_taken_agents = np.random.default_rng().integers(
                0, len(population), size=2)

            taken_agents_to_fight_for_life = [
                population[i] for i in random_taken_agents]

            random_number = np.random.uniform(0, 1)
            fitness1 = taken_agents_to_fight_for_life[0].compute_performance()
            fitness2 = taken_agents_to_fight_for_life[1].compute_performance()

            if random_number < threshold:
                if fitness1 > fitness2:
                    selected_agents.append(taken_agents_to_fight_for_life[0])
                else:
                    selected_agents.append(taken_agents_to_fight_for_life[1])
            else:
                if fitness1 > fitness2:
                    selected_agents.append(taken_agents_to_fight_for_life[1])
                else:
                    selected_agents.append(taken_agents_to_fight_for_life[0])


class DeterministicTournament(Selection):
    """
        Tournament selection method
    """

    def select(self, population: list[Agent], K: int, **kargs) -> list[Agent]:
        """
            Selects the individuals from the population
        """

        if kargs.get["M"] is None:
            raise Exception("M is required")

        M = kargs.get["M"]

        selected_agents = []

        for _ in range(K):
            random_taken_agents = np.random.default_rng().integers(
                0, len(population), size=M)  # TODO: SE PUEDE REPETIR:?????????

            taken_agents = [population[i] for i in random_taken_agents]

            taken_agents_fitness = np.array(
                agent.compute_performance() for agent in taken_agents)

            best_agent_index = np.argmax(taken_agents_fitness)

            selected_agents.append(taken_agents[best_agent_index])

# 0.4, 0.5, 1 population
# 1, 0.5, 0.4 ordered
# 0.9 0.7 0.5 fitness per agent


class Ranking(Selection):
    """
        Ranking selection method
    """

    def select(self, population: list[Agent], K: int) -> list[Agent]:
        """
            Selects the individuals from the population
        """
        ordered_by_fitness = population.sort(
            key=lambda agent: agent.compute_performance())

        population_lenght = len(population)
        fitness_per_agent = []

        for rank in range(1, population_lenght + 1):
            fitness_per_agent.append(
                (population_lenght - rank) / population_lenght)

        return Roulette().roulette_with_give_fitness(ordered_by_fitness, K, fitness_per_agent)


class SelectionOptions(Enum):
    """
        Selection options enum
    """
    ELITE = "Elite"
    ROULETTE = "Roulette"
    UNIVERSAL = "Universal"
    BOLTZMANN = "Boltzmann"
    DETERMINISTIC_TOURNAMENT = "DeterministicTournament"
    PROBABILISTIC_TOURNAMENT = "ProbabilisticTournament"
    RANKING = "Ranking"

    @classmethod
    def from_string(cls, name: str) -> Optional['SelectionOptions']:
        """
            Returns the enum value from the string
        """
        try:
            return SelectionOptions(name)
        except ValueError:
            return None

    @classmethod
    def get_instance_from_name(cls, name: str) -> Optional[Selection]:
        """
            Returns enum instance from the string
        """
        try:
            return SelectionOptions.get_instance_from_type(SelectionOptions(name))
        except ValueError:
            return None

    @classmethod
    def get_instance_from_type(cls, role_type: 'SelectionOptions') -> Optional[Selection]:
        """
            Returns enum instance from the enum
        """
        selected_class = globals()[role_type.value]
        return selected_class()
