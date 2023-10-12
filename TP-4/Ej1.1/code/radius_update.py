from abc import ABC, abstractmethod


class RadiusUpdate(ABC):
    @abstractmethod
    def update(self, original_radius: float, iteration: int):
        pass


class IdentityUpdate(RadiusUpdate):

    def update(self, original_radius: float, iteration: int):
        return original_radius


class ProgressiveReduction(RadiusUpdate):

    def update(self, original_radius: float, iteration: int):
        if iteration == 0:
            return original_radius
        return max(1 / iteration, 1)