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
            :key n (int): Population size.
            :key crossover_method (Crossover): Crossover method for agent breeding
            :key selections (List[Selection]): Selection methods for agent filtering
            :key mutation (Mutation): Mutation function for agents.
            :key selection_strategy (str): The selection strategy parsed from a string.
            :key crossover_proportion (float): Proportion of agents used for parent_to_cross selection.
            :key selection_proportion (float): Proportion of agents used to select in replacement.
            :key k (int): Amount of children to generate in each iteration.
            :key role (Role): The role to test in the simulation.

            :key max_iterations (int): Maximum amount of iterations to run the simulation.
            :key max_generations_without_improvement (int): Maximum amount of generations without improvement to run the simulation.


        :raises ValueError: If selection_proportion is not within the range [0, 1].
        """
        self.n = kwargs["n"]
        self.population: List[Agent] = self.generate_gen_0(1.3, 2.0)
        self.crossover_method: Crossover = kwargs["crossover"]
        self.selections: List[Selection] = kwargs["selections"]
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
        parents_to_cross = self.select_parents_to_cross()
        children = self.crossover(parents_to_cross)
        children = self.mutation(children)
        self.population = self.replacement(children, self.population)

        # Reemplazo, quién te conoce??
    def select_parents_to_cross(self):
        parents_to_cross: list[Agent] = []

        # Get populations
        k1 = int(self.k * self.crossover_proportion)
        k2 = self.k - k1
        population_amount: int = len(self.population)
        population_1_amount: int = int(population_amount * self.crossover_proportion)

        population_1 = self.population[0:population_1_amount]
        population_2 = self.population[population_1_amount:population_amount]

        # Select parents to cross
        parents_to_cross_1 = self.selections[0].select(population_1, k1)
        parents_to_cross = parents_to_cross + parents_to_cross_1

        parents_to_cross = parents_to_cross + \
            self.selections[1].select(population_2, k2)

        return parents_to_cross

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

    def selection_replacement(self, population_to_select: list[Agent], amount_to_select: int):
        pop_amount = len(population_to_select)
        amount_to_select_1 = int(amount_to_select * self.selection_proportion)
        amount_to_select_2 = amount_to_select_1 - amount_to_select_1
        pop_1_amount: int = int(pop_amount * self.selection_proportion)

        pop_1 = population_to_select[0:pop_1_amount]
        pop_2 = population_to_select[pop_1_amount:pop_amount]

        return self.selections[2].select(pop_1, amount_to_select_1,
                                         T=self.bolzmann_temperature, M=self.deterministic_tournament_m,
                                         Threshold=self.probabilistic_tournament_threshold) + \
            self.selections[3].select(pop_2, amount_to_select_2,
                                      T=self.bolzmann_temperature, M=self.deterministic_tournament_m,
                                      Threshold=self.probabilistic_tournament_threshold)

    def replacement(self, children: List[Agent], parents: List[Agent]) -> List[Agent]:
        if self.selection_strategy == SelectionStrategy.TRADITIONAL:
            population_to_select = children + parents
            return self.selection_replacement(population_to_select, self.n)

        elif self.selection_strategy == SelectionStrategy.YOUNG_BIAS:
            children_amount = len(children)
            if children_amount >= self.n:
                return self.selection_replacement(children, self.n)

            parents_selected = self.selection_replacement(parents, self.n - len(children))
            return children + parents_selected
        else:
            raise "WTF"

    def crossover(self, parents_to_cross: list[Agent]):
        # Cross populations
        children: List[Agent] = []
        children = children + \
            self.__crossover_with_method(self.crossover_method.cross, parents_to_cross)
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
