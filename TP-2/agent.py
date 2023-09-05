from typing import Sequence

from role import CharacterStats, Role, ItemStats, Modifiers, Cromosome, get_attack, get_defense


class Agent:
    role: Role
    stats: CharacterStats
    height: float
    cromosome: Sequence


    def __init__(self, role: Role, cromosome: Cromosome):
        self.cromosome = cromosome.as_list
        # items = ItemStats.from_weights(cromosome[0:-1])
        self.height = cromosome.height
        self.role = role
        self.stats = CharacterStats(cromosome.stats)

    def compute_performance(self) -> float:
        """
            fitness function
        """
        modifiers: Modifiers = Modifiers(self.height)
        attack: float = get_attack(self.stats, modifiers)
        defense: float = get_defense(self.stats, modifiers)
        return self.role.compute_performance(attack, defense)

    def __str__(self):
        return f"Agent({self.role.name}, {self.height}, {self.stats})"

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Agent):
            return False
        return self.role == __value.role and self.height == __value.height and self.stats == __value.stats
