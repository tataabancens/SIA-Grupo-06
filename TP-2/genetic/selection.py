from enum import Enum
from abc import ABC, abstractmethod
from typing import Optional


class Selection(ABC):
    """
        Selection method abstract class
    """

    @abstractmethod
    def select(self, population, K):
        """
            Selects the individuals from the population
        """
        pass


class Elite(Selection):
    """
        Elite selection method
    """

    def select(self, population, K):
        """
            Selects the individuals from the population
        """
        pass


class Roulette(Selection):
    """
        Roulette selection method
    """

    def select(self, population, K):
        """
            Selects the individuals from the population
        """
        pass


class Universal(Selection):
    """
        Universal selection method
    """

    def select(self, population, K):
        """
            Selects the individuals from the population
        """
        pass


class Boltzmann(Selection):
    """
        Boltzmann selection method
    """

    def select(self, population, K):
        """
            Selects the individuals from the population
        """
        pass


class Tournament(Selection):
    """
        Tournament selection method
    """

    def select(self, population, K):
        """
            Selects the individuals from the population
        """
        pass


class Ranking(Selection):
    """
        Ranking selection method
    """

    def select(self, population, K):
        """
            Selects the individuals from the population
        """
        pass


class SelectionOptions(Enum):
    """
        Selection options enum
    """
    ELITE = "Elite"
    ROULETTE = "Roulette"
    UNIVERSAL = "Universal"
    BOLTZMANN = "Boltzmann"
    TOURNAMENT = "Tournament"
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
