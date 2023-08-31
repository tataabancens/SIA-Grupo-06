from enum import Enum
from abc import ABC, abstractmethod
from typing import Optional
from numpy import random, concatenate
from agent import Agent
from role import RoleType


class Crossover(ABC):
    """
        Crossover method abstract class
    """

    @abstractmethod
    def cross(self, parents: tuple[Agent]) -> tuple[Agent]:
        """
            Crossover the individuals from the population
        """
        pass


class OnePoint(Crossover):
    """
        One point crossover method
    """

    def cross(self, parents: tuple[Agent]) -> tuple[Agent]:
        """
            Crossover the individuals from the population
        """

        s = len(parents[0].cromosome)
        p = random.default_rng().integers(0, s)

        children_cromosomes = []
        children_cromosomes[0] = concatenate(
            (parents[0].cromosome[0:p], parents[1].cromosome[p:s]), axis=0)
        children_cromosomes[1] = concatenate(
            (parents[1].cromosome[0:p], parents[0].cromosome[p:s]), axis=0)

        # TODO: Create children agents
        return (
            Agent(role=RoleType.get_instance_from_name("Defence"),
                  cromosome=children_cromosomes[0]),
            Agent(role=RoleType.get_instance_from_name("Defence"),
                  cromosome=children_cromosomes[0])
        )


class TwoPoint(Crossover):
    """
        Two point crossover method
    """

    def cross(self, parents: tuple[Agent]) -> tuple[Agent]:
        """
            Crossover the individuals from the population
        """
        s = len(parents[0].cromosome)
        p_1, p_2 = random.default_rng().integers(0, s, size=2)

        children_cromosomes = []
        children_cromosomes[0] = concatenate(
            (parents[0].cromosome[0:p_1], parents[1].cromosome[p_1:p_2]), axis=0)
        children_cromosomes[1] = concatenate(
            (parents[1].cromosome[0:p_1], parents[0].cromosome[p_1:p_2]), axis=0)

        # TODO: Create children agents
        return (
            Agent(role=RoleType.get_instance_from_name("Defence"),
                  cromosome=children_cromosomes[0]),
            Agent(role=RoleType.get_instance_from_name("Defence"),
                  cromosome=children_cromosomes[0])
        )


class Uniform(Crossover):
    """
        Uniform crossover method
    """

    def cross(self, parents: tuple[Agent]) -> tuple[Agent]:
        """
            Crossover the individuals from the population
        """
        p_s = random.default_rng().uniform(
            0, 1, size=len(parents[0].cromosome))

        children_cromosomes = []
        for i in range(len(parents[0].cromosome)):
            if p_s[i] < 0.5:
                children_cromosomes[0][i] = parents[0].cromosome[i]
                children_cromosomes[1][i] = parents[1].cromosome[i]
            else:
                children_cromosomes[0][i] = parents[1].cromosome[i]
                children_cromosomes[1][i] = parents[0].cromosome[i]


class Anular(Crossover):
    """
        Anular crossover method
    """

    def cross(self, parents: tuple[Agent]) -> tuple[Agent]:
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
