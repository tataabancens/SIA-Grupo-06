from typing import Sequence

from role import CharacterStats, Role, ItemStats, Modifiers, Chromosome, get_attack, get_defense


_MIN_HEIGHT = 1.3
_MAX_HEIGHT = 2.0


class Agent:
    role: Role
    stats: CharacterStats
    height: float  # Entre 0 y 1
    chromosome: Sequence

    def __init__(self, role: Role, chromosome: Chromosome):
        self.chromosome = chromosome.as_list
        # items = ItemStats.from_weights(chromosome[0:-1])
        self.height = chromosome.height
        self.role = role
        self.stats = CharacterStats(chromosome.stats)

    def __repr__(self):
        return f"Agent perf: {self.compute_performance()}"

    def compute_performance(self) -> float:
        """
            fitness function
        """
        modifiers: Modifiers = Modifiers(self.get_height())
        attack: float = get_attack(self.stats, modifiers)
        defense: float = get_defense(self.stats, modifiers)
        return self.role.compute_performance(attack, defense)

    def get_height(self) -> float:
        """
            Returns agent height in meters
        """
        return self.height * (_MAX_HEIGHT - _MIN_HEIGHT) + _MIN_HEIGHT

    def __str__(self):
        return f"Agent(role={self.role}, height={self.get_height()}, stats={self.stats})"

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Agent):
            return False
        return self.role == __value.role and self.height == __value.height and self.stats == __value.stats
