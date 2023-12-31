from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from enum import Enum
from typing import Optional, Sequence
from partition import normalize_partition


@dataclass
class Stats:
    strength: float  # Fuerza
    agility: float  # Agilidad
    proficiency: float  # Pericia
    toughness: float  # Resistencia
    health: float  # Vida

    def get_as_list(self):
        return [self.strength, self.agility, self.proficiency, self.toughness, self.health]


@dataclass
class ItemStats(Stats):
    __target: float = field(init=False, default=150.0)

    def __post_init__(self) -> None:
        stats_sum: float = self.strength + self.agility + \
                           self.proficiency + self.toughness + self.health
        if not math.isclose(stats_sum, self.__target, abs_tol=0.01):
            raise f"Item stats do not sum up to target of {self.__target}. Sum is {stats_sum}"

    @classmethod
    def from_weights(cls, weights: list) -> 'ItemStats':
        final_values = normalize_partition(weights, ItemStats.__target)
        return ItemStats(
            strength=final_values[0],
            agility=final_values[1],
            proficiency=final_values[2],
            toughness=final_values[3],
            health=final_values[4]
        )


@dataclass
class Chromosome:
    stats: ItemStats
    height: float
    as_list: Sequence = field(init=False)

    def __init__(self, stats: ItemStats, height: float):
        self.height = height
        self.stats = stats
        self.as_list = [stats.strength, stats.agility, stats.proficiency, stats.toughness, stats.health, self.height]


    @classmethod
    def from_list(cls, chromosome: Sequence):
        return cls(ItemStats(chromosome[0], chromosome[1], chromosome[2], chromosome[3], chromosome[4]), chromosome[5])

    @classmethod
    def from_unnormalized_list(cls, chromosome: Sequence):
        stats = ItemStats.from_weights(chromosome[:-1])
        height = chromosome[-1]
        if height > 1:
            height = 1
        if height < 0:
            height = 0
        return cls(stats, height)


class CharacterStats(Stats):

    def __init__(self, items: ItemStats):
        self.strength = 100 * math.tanh(0.01 * items.strength)
        self.agility = math.tanh(0.01 * items.agility)
        self.proficiency = 0.6 * math.tanh(0.01 * items.proficiency)
        self.toughness = math.tanh(0.01 * items.toughness)
        self.health = 100 * math.tanh(0.01 * items.health)

    def __eq__(self, __value: object) -> bool:
        return super().__eq__(__value)


class Role(ABC):
    @abstractmethod
    def compute_performance(self, attack: float, defense: float) -> float:
        raise NotImplementedError()

    def __eq__(self, __value: object) -> bool:
        return super().__eq__(__value)


class Fighter(Role):
    def compute_performance(self, attack: float, defense: float) -> float:
        return 0.6 * attack + 0.4 * defense

    def __str__(self):
        return "Fighter"


class Archer(Role):
    def compute_performance(self, attack: float, defense: float) -> float:
        return 0.9 * attack + 0.1 * defense

    def __str__(self):
        return "Archer"


class Defender(Role):
    def compute_performance(self, attack: float, defense: float) -> float:
        return 0.1 * attack + 0.9 * defense

    def __str__(self):
        return "Defender"


class Infiltrate(Role):
    def compute_performance(self, attack: float, defense: float) -> float:
        return 0.8 * attack + 0.3 * defense

    def __str__(self):
        return "Infiltrate"


class RoleType(Enum):
    FIGHTER = "Fighter"
    ARCHER = "Archer"
    DEFENDER = "Defender"
    INFILTRATE = "Infiltrate"

    @classmethod
    def from_string(cls, name: str) -> Optional['RoleType']:
        try:
            return RoleType(name)
        except ValueError:
            return None

    @classmethod
    def get_instance_from_name(cls, name: str) -> Optional[Role]:
        try:
            return RoleType.get_instance_from_type(RoleType(name))
        except ValueError:
            return None

    @classmethod
    def get_instance_from_type(cls, role_type: 'RoleType') -> Optional[Role]:
        selected_class = globals()[role_type.value]
        return selected_class()


class Modifiers:
    attack: float
    defense: float

    def __init__(self, height: float):
        if not (1.3 <= height <= 2.0):
            self.attack = 1
            self.defense = 1
            return
        fourth_pow_term = (3 * height - 5) ** 4
        second_pow_term = (3 * height - 5) ** 2
        half_height = height / 2
        self.attack = 0.5 - fourth_pow_term + second_pow_term + half_height
        self.defense = 2 + fourth_pow_term - second_pow_term - half_height


def get_attack(stats: CharacterStats, modifiers: Modifiers):
    return (stats.agility + stats.proficiency) * stats.strength * modifiers.attack


def get_defense(stats: CharacterStats, modifiers: Modifiers):
    return (stats.toughness + stats.proficiency) * stats.health * modifiers.defense
