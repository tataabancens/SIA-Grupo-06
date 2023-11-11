class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input: list[float]):
        # TODO: return output
        pass

    def backward(self, output_gradient: list[float], learning_rate: float):
        # TODO: update parameters and return input gradient
        pass
