from enum import Enum
from abc import ABC, abstractmethod
from typing import Optional


class Crossover(ABC):
    """
        Crossover method abstract class
    """

    @abstractmethod
    def cross(self, parents):
        """
            Crossover the individuals from the population
        """
        pass


class OnePoint(Crossover):
    """
        One point crossover method
    """

    def cross(self, parents):
        """
            Crossover the individuals from the population
        """
        pass


class TwoPoint(Crossover):
    """
        Two point crossover method
    """

    def cross(self, parents):
        """
            Crossover the individuals from the population
        """
        pass


class Uniform(Crossover):
    """
        Uniform crossover method
    """

    def cross(self, parents):
        """
            Crossover the individuals from the population
        """
        pass


class Anular(Crossover):
    """
        Anular crossover method
    """

    def cross(self, parents):
        """
            Crossover the individuals from the population
        """
        pass


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
