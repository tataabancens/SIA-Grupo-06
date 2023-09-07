from enum import Enum
from abc import ABC, abstractmethod
from typing import Optional
from numpy import random, concatenate
from agent import Agent
from role import RoleType, Chromosome


class Crossover(ABC):
    """
        Crossover method abstract class
    """

    @abstractmethod
    def cross(self, parents: tuple[Agent, Agent]) -> tuple[Agent, Agent]:
        """
            Crossover the individuals from the population
        """
        pass


class OnePoint(Crossover):
    """
        One point crossover method
    """

    @classmethod
    def cross(cls, parents: tuple[Agent, Agent]) -> tuple[Agent, Agent]:
        """
            Crossover the individuals from the population
        """

        s = len(parents[0].chromosome)
        p = random.default_rng().integers(s)

        children_chromosome_1 = concatenate(
            (parents[0].chromosome[0:p], parents[1].chromosome[p:s]), axis=0)
        children_chromosome_2 = concatenate(
            (parents[1].chromosome[0:p], parents[0].chromosome[p:s]), axis=0)
        role = parents[0].role
        return (
            Agent(role=role,
                  chromosome=Chromosome.from_unnormalized_list(children_chromosome_1)),

            Agent(role=role,
                  chromosome=Chromosome.from_unnormalized_list(children_chromosome_2))
        )


class TwoPoint(Crossover):
    """
        Two point crossover method
    """

    @classmethod
    def cross(cls, parents: tuple[Agent, Agent]) -> tuple[Agent, Agent]:
        """
            Crossover the individuals from the population
        """
        s = len(parents[0].chromosome)
        p_1, p_2 = random.default_rng().integers(0, s, size=2)
        if p_1 > p_2:
            p_1, p_2 = p_2, p_1

        children_chromosomes = [
            concatenate((parents[0].chromosome[0:p_1], parents[1].chromosome[p_1:p_2], parents[0].chromosome[p_2:s]), axis=0),
            concatenate((parents[1].chromosome[0:p_1], parents[0].chromosome[p_1:p_2], parents[1].chromosome[p_2:s]), axis=0)
        ]

        role = parents[0].role
        return (
            Agent(role=role,
                  chromosome=Chromosome.from_unnormalized_list(children_chromosomes[0])),
            Agent(role=role,
                  chromosome=Chromosome.from_unnormalized_list(children_chromosomes[1]))
        )


class Uniform(Crossover):
    """
        Uniform crossover method
    """
    @classmethod
    def cross(cls, parents: tuple[Agent, Agent]) -> tuple[Agent, Agent]:
        """
            Crossover the individuals from the population
        """
        p_s = random.default_rng().uniform(
            0, 1, size=len(parents[0].chromosome))

        children_chromosomes = []
        for i in range(len(parents[0].chromosome)):
            if p_s[i] < 0.5:
                children_chromosomes[0][i] = parents[0].chromosome[i]
                children_chromosomes[1][i] = parents[1].chromosome[i]
            else:
                children_chromosomes[0][i] = parents[1].chromosome[i]
                children_chromosomes[1][i] = parents[0].chromosome[i]

        role = parents[0].role
        return (
            Agent(role=role,
                  chromosome=Chromosome.from_unnormalized_list(children_chromosomes[0])),
            Agent(role=role,
                  chromosome=Chromosome.from_unnormalized_list(children_chromosomes[1]))
        )


class Anular(Crossover):
    """
        Anular crossover method
    """

    @classmethod
    def cross(cls, parents: tuple[Agent, Agent]) -> tuple[Agent, Agent]:
        """
            Crossover the individuals from the population
        """
        return TwoPoint().cross(parents)  # 🤐🤐🤐🤐🤐


class CrossoverOptions(Enum):
    """
        Selection options enum
    """
    ONE_POINT = "OnePoint"
    TWO_POINT = "TwoPoint"
    UNIFORM = "Uniform"
    ANULAR = "Anular"

    @classmethod
    def from_string(cls, name: str) -> Optional['CrossoverOptions']:
        """
            Returns the enum value from the string
        """
        try:
            return CrossoverOptions(name)
        except ValueError:
            return None

    @classmethod
    def get_instance_from_name(cls, name: str) -> Optional[Crossover]:
        """
            Returns enum instance from the string
        """
        try:
            return CrossoverOptions.get_instance_from_type(CrossoverOptions(name))
        except ValueError:
            return None

    @classmethod
    def get_instance_from_type(cls, role_type: 'CrossoverOptions') -> Optional[Crossover]:
        """
            Returns enum instance from the enum
        """
        selected_class = globals()[role_type.value]
        return selected_class()


if __name__ == "__main__":
    method = CrossoverOptions.get_instance_from_name("OnePoint")
