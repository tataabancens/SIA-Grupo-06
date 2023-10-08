from typing import List


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input: List[float]):
        # TODO: return output
        pass

    def backward(self, output_gradient: List[float], learning_rate: float):
        # TODO: update parameters and return input gradient
        pass
