from role import CharacterStats, Role, ItemStats, Modifiers, get_attack, get_defense


class Agent:
    role: Role
    stats: CharacterStats
    height: float
    cromosome: list[float]

    def __init__(self, role: Role, cromosome: list):
        self.cromosome = cromosome
        items = ItemStats.from_weights(cromosome[0:-1])
        self.height = cromosome[-1]
        self.role = role
        self.stats = CharacterStats(items)

    def compute_performance(self) -> float:
        """
            fitness function
        """
        modifiers: Modifiers = Modifiers(self.height)
        attack: float = get_attack(self.stats, modifiers)
        defense: float = get_defense(self.stats, modifiers)
        return self.role.compute_performance(attack, defense)
