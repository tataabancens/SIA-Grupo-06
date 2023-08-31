from typing import List, Tuple, Callable
from agent import Agent
from role import Role
from abc import ABC, abstractmethod
from enum import Enum


# class SelectionStrategy(ABC):
#
#     @abstractmethod
#     def merge_population(self, children: List[Agent], parents: List[Agent]) -> List[Agent]:
#         raise NotImplementedError()
#
#
# class YoungBias(SelectionStrategy):
#
#     def __init__(self, k: int):
#         self.k = k
#
#     def merge_population(self, children: List[Agent], parents: List[Agent]) -> List[Agent]:
#         if len(children) > self.k:
#             return children[0:self.k]
#         return children[0:self.k] + parents[]


class SelectionStrategy(Enum):
    YOUNG_BIAS = "young"
    TRADITIONAL = "traditional"


class Simulation:

    def __init__(self, *args, **kwargs):
        self.population: List[Agent] = kwargs["gen_0"]
        self.crossovers: Tuple[Callable[[Agent, Agent], Tuple[Agent, Agent]]] = kwargs["crossovers"]
        self.selections: Tuple[Callable[[List[Agent], int], List[Agent]]] = kwargs["selections"]
        self.mutation: Callable[[List[Agent]], List[Agent]] = kwargs["mutation"]
        # The parameter is a string that is parsed here
        self.selection_strategy: SelectionStrategy = SelectionStrategy(kwargs["selection_strategy"])
        self.crossover_proportion: float = kwargs["crossover_proportion"]
        self.selection_proportion: float = kwargs["selection_proportion"]
        if not (0 <= self.selection_proportion <= 1):
            raise ValueError("Selection proportion is not in the range [0,1]")
        self.k: int = kwargs["k"]
        self.role = kwargs["role"]

    def end_condition(self) -> bool:
        return False

    def run(self):
        while not self.end_condition():
            self.iterate()

    def iterate(self):
        children = self.crossover(self.crossover_proportion)
        children = self.mutation(children)
        self.population = self.population = self.selection(children, self.population)
        # Reemplazo, quién te conoce??

    def selection(self, children: List[Agent], parents: List[Agent]) -> List[Agent]:
        # N -> individuos de la población
        # K -> individuos a seleccionar
        pop_size: int = len(self.population)
        selected = []
        method_a_proportion = int(self.k * self.selection_proportion)
        population_to_select = []

        if self.selection_strategy == SelectionStrategy.TRADITIONAL:
            population_to_select = children + parents

        elif self.selection_strategy == SelectionStrategy.YOUNG_BIAS:
            children_amount = len(children)
            if children_amount >= self.k:
                return children

            selected = children
            parents_to_select_amount: int = pop_size - children_amount
            method_a_proportion = int(parents_to_select_amount * self.selection_proportion)
            selected = selected + self.selections[0](parents, method_a_proportion)
            selected = selected + self.selections[1](parents, parents_to_select_amount - method_a_proportion)
            return selected
        else:
            raise "WTF"

        selected = selected + self.selections[0](population_to_select, method_a_proportion)
        selected = selected + self.selections[1](population_to_select, self.k - method_a_proportion)

        return selected

    def crossover(self, method_a_proportion: float):
        # Get populations
        population_amount: int = len(self.population)
        a_population_amount: int = int(population_amount * method_a_proportion)
        b_population_amount: int = population_amount - a_population_amount
        a_population = self.population[0:a_population_amount]
        b_population = self.population[0:b_population_amount]

        # Cross populations
        children: List[Agent] = []
        children = children + self.__crossover_with_method(self.crossovers[0], a_population)
        children = children + self.__crossover_with_method(self.crossovers[1], b_population)

        return children

    @staticmethod
    def __crossover_with_method(method: Callable[[Agent, Agent], Tuple[Agent, Agent]], population: List[Agent]) -> List[
        Agent]:
        children: List[Agent] = []
        population_amount = len(population)
        for i in range(1, population_amount, 2):
            children = children + list(method(population[-1], population[0]))

        if population_amount % 2 != 0:
            children = children + list(method(population[-1], population[0]))
        return children

    @staticmethod
    def __selection_with_method(method: Callable[[List[Agent], Int], List[Agent]], population: List[Agent],k :int)->List[Agent]:
        return method(population, k)


def main():
    print("Hello world")


if __name__ == '__main__':
    main()
