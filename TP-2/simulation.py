import random
from typing import List, Tuple, Callable
from agent import Agent
from genetic.mutation import MutationOptions, Mutation
from partition import random_partition
from role import Role, RoleType, Cromosome, ItemStats
from genetic.selection import SelectionStrategy, SelectionOptions, Selection
from genetic.crossover import CrossoverOptions, Crossover


class Simulation:

    iteration = 0
    iteration_without_improvement = 0
    iteration_max_performance = 0

    def __init__(self, *args, **kwargs):
        """
        Method description

        :param args: Variable length argument list.
        :param kwargs: Arbitrary keyword arguments.
            :key gen_0 (List[Agent]): Initial population of agents
            :key crossovers (Tuple[Crossover]): Crossover methods for agent breeding
            :key selections (Tuple[Selection]): Selection methods for agent filtering
            :key mutation (Mutation): Mutation function for agents.
            :key selection_strategy (str): The selection strategy parsed from a string.
            :key crossover_proportion (float): Proportion of agents used for crossovers.
            :key selection_proportion (float): Proportion of agents retained after selection.
            :key k (int): Amount of agents to select in each iteration.
            :key role (Role): The role to test in the simulation.

            :key max_iterations (int): Maximum amount of iterations to run the simulation.
            :key max_generations_without_improvement (int): Maximum amount of generations without improvement to run the simulation.


        :raises ValueError: If selection_proportion is not within the range [0, 1].
        """
        self.n = kwargs["n"]
        self.population: List[Agent] = self.generate_gen_0(1.3, 2.0)
        self.crossovers: Tuple[Crossover] = kwargs["crossovers"]
        self.selections: Tuple[Selection] = kwargs["selections"]
        self.mutation_method: Mutation = kwargs["mutation"]
        # The parameter is a string that is parsed here
        self.selection_strategy: SelectionStrategy = SelectionStrategy(
            kwargs["selection_strategy"])
        self.crossover_proportion: float = kwargs["crossover_proportion"]
        self.selection_proportion: float = kwargs["selection_proportion"]
        if not (0 <= self.selection_proportion <= 1):
            raise ValueError("Selection proportion is not in the range [0,1]")
        self.k: int = kwargs["k"]
        self.bolzmann_temperature: float = kwargs["bolzmann_temperature"]
        self.deterministic_tournament_m: int = kwargs["deterministic_tournament_m"]
        self.probabilistic_tournament_threshold: float = kwargs["probabilistic_tournament_threshold"]
        self.role: Role = kwargs["role"]
        self.max_iterations: int = kwargs["max_iterations"]
        self.max_generations_without_improvement: int = kwargs["max_generations_without_improvement"]

    def end_condition(self) -> bool:
        if self.iteration >= self.max_iterations:
            return True

        if self.iteration_without_improvement >= self.max_generations_without_improvement:
            return True

        self.population.sort(key=lambda agent: agent.compute_performance())
        max_performance = self.population[0]

        # TODO: Sacar esto de aca
        print(max_performance.compute_performance())

        if self.iteration_max_performance == max_performance:
            self.iteration_without_improvement += 1
        else:
            self.iteration_without_improvement = 0
            self.iteration_max_performance = max_performance

        return False

    def run(self):
        while not self.end_condition():
            self.iterate()
            self.iteration += 1

    def iterate(self):
        children = self.crossover(self.crossover_proportion)
        children = self.mutation(children)
        self.population = self.population = self.selection(
            children, self.population)

        # Reemplazo, quién te conoce??

    def mutation(self, children: List[Agent]) -> List[Agent]:
        for child in children:
            gens_mutated: list[int] | None = self.mutation_method.mutate(
                child, 0.1)
            if gens_mutated:
                for gen in gens_mutated:
                    if gen != 5:
                        child.cromosome = Cromosome.from_unnormalized_list(
                            child.cromosome).as_list
                        break
        return children

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
            method_a_proportion = int(
                parents_to_select_amount * self.selection_proportion)
            selected = selected + \
                self.selections[0].select(
                    parents, method_a_proportion, T=self.bolzmann_temperature, M=self.deterministic_tournament_m, Threshold=self.probabilistic_tournament_threshold)
            selected = selected + \
                self.selections[1].select(
                    parents, parents_to_select_amount - method_a_proportion, T=self.bolzmann_temperature, M=self.deterministic_tournament_m, Threshold=self.probabilistic_tournament_threshold)
            return selected
        else:
            raise "WTF"

        selected = selected + \
            self.selections[0].select(
                population_to_select, method_a_proportion, T=self.bolzmann_temperature, M=self.deterministic_tournament_m, Threshold=self.probabilistic_tournament_threshold)
        selected = selected + \
            self.selections[1].select(population_to_select,
                                      self.k - method_a_proportion, T=self.bolzmann_temperature, M=self.deterministic_tournament_m, Threshold=self.probabilistic_tournament_threshold)

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
        children = children + \
            self.__crossover_with_method(
                self.crossovers[0].cross, a_population)
        children = children + \
            self.__crossover_with_method(
                self.crossovers[1].cross, b_population)

        return children

    @staticmethod
    def __crossover_with_method(method: Callable[[Tuple[Agent, Agent]], Tuple[Agent, Agent]], population: List[Agent]) -> List[
            Agent]:
        children: List[Agent] = []
        population_amount = len(population)
        for i in range(1, population_amount, 2):
            parents = (population[i-1], population[i])
            children = children + list(method(parents))

        if population_amount % 2 != 0:
            parents = (population[-1], population[0])
            children = children + list(method(parents))
        return children

    @staticmethod
    def __selection_with_method(method: Callable[[List[Agent], int], List[Agent]], population: List[Agent], k: int) -> List[Agent]:
        return method(population, k)

    def generate_gen_0(self, min_height: float, max_height: float) -> list[Agent]:
        agents: list[Agent] = []
        for _ in range(self.n):
            partition = random_partition(150, 5)

            items = ItemStats(
                strength=partition[0],
                agility=partition[1],
                proficiency=partition[2],
                toughness=partition[3],
                health=partition[4])

            random_height = random.uniform(min_height, max_height)
            role = RoleType.get_instance_from_name("Fighter")
            cromosome = Cromosome(items, random_height)
            agents.append(Agent(role, cromosome))
        return agents


def main():
    n = 10
    crossovers = (CrossoverOptions.get_instance_from_name("OnePoint"),
                  CrossoverOptions.get_instance_from_name("TwoPoint"))

    selections = (SelectionOptions.get_instance_from_name("DeterministicTournament"),
                  SelectionOptions.get_instance_from_name("ProbabilisticTournament"))

    mutation = MutationOptions.get_instance_from_name("OneGen")
    selection_strategy = SelectionStrategy.TRADITIONAL
    a, b = 0.5, 0.5
    max_iter, max_iter_without, k = 2000, 5, 20

    bolzmann_temperature = 0.5
    deterministic_tournament_m = 5
    probabilistic_tournament_threshold = 0.5

    role = RoleType.get_instance_from_name("Fighter")
    simulation = Simulation(n=n, crossovers=crossovers, selections=selections,
                            mutation=mutation, selection_strategy=selection_strategy,
                            crossover_proportion=a, selection_proportion=b, k=k, role=role,
                            bolzmann_temperature=bolzmann_temperature,
                            deterministic_tournament_m=deterministic_tournament_m,
                            probabilistic_tournament_threshold=probabilistic_tournament_threshold,
                            max_iterations=max_iter, max_generations_without_improvement=max_iter_without)
    simulation.run()


if __name__ == '__main__':
    main()
