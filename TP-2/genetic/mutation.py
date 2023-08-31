from enum import Enum
from abc import ABC, abstractmethod
from typing import Optional


class Mutation(ABC):
    """
        Mutation method abstract class
    """

    @abstractmethod
    def mutate(self, population, p_m):
        """
            Mutate the individuals from the population
        """
        pass


class Gen(Mutation):
    """
        Se altera un solo gen con una probabilidad P_m
    """

    def mutate(self, population, p_m):
        """
            Mutate the individuals from the population
        """
        pass


class LimitedMultiGen(Mutation):
    """
        Se selecciona una cantidad [1,M] (azarosa) de genes para mutar, con probabilidad P_m
    """

    def mutate(self, population, p_m):
        """
            Mutate the individuals from the population
        """
        pass


class UniformMultiGen(Mutation):
    """
         Cada gen tiene una probabilidad P_m de ser mutado
    """

    def mutate(self, population, p_m):
        """
            Mutate the individuals from the population
        """
        pass


class Complete(Mutation):
    """
        Con una probabilidad P_m se mutan todos los genes del individuo, acorde a la función de mutación definida para cada gen
    """

    def mutate(self, population, p_m):
        """
            Mutate the individuals from the population
        """
        pass


class MutationOptions(Enum):
    """
        Selection options enum
    """
    GEN = "Gen"
    LIMITED_MULTI_GEN = "LimitedMultiGen"
    UNIFORM_MULTI_GEN = "UniformMultiGen"
    COMPLETE = "Complete"

    @classmethod
    def from_string(cls, name: str) -> Optional['MutationOptions']:
        """
            Returns the enum value from the string
        """
        try:
            return MutationOptions(name)
        except ValueError:
            return None

    @classmethod
    def get_instance_from_name(cls, name: str) -> Optional[Mutation]:
        """
            Returns enum instance from the string
        """
        try:
            return MutationOptions.get_instance_from_type(MutationOptions(name))
        except ValueError:
            return None

    @classmethod
    def get_instance_from_type(cls, role_type: 'MutationOptions') -> Optional[Mutation]:
        """
            Returns enum instance from the enum
        """
        selected_class = globals()[role_type.value]
        return selected_class()
