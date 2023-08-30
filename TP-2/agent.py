from role import CharacterStats, Role, ItemStats, Modifiers, get_attack, get_defense


class Agent:
    role: Role
    stats: CharacterStats

    def __init__(self, role: Role, items: ItemStats):
        self.role = role
        self.stats = CharacterStats(items)

    def compute_performance(self, height: float) -> float:
        modifiers: Modifiers = Modifiers(height)
        attack: float = get_attack(self.stats, modifiers)
        defense: float = get_defense(self.stats, modifiers)
        return self.role.compute_performance(attack, defense)
