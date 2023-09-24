import numpy as np


class Optimizer:
    def __init__(self):
        pass

    def adjust(self, learning_rate, gradient):
        pass

    def get_one(self) -> 'Optimizer':
        pass


class GradientDescent(Optimizer):
    def __init__(self):
        super().__init__()

    def adjust(self, gradient, learning_rate):
        return learning_rate * gradient

    def get_one(self):
        return GradientDescent()


class Adam(Optimizer):
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def get_one(self):
        return Adam(self.beta1, self.beta2, self.epsilon)

    def adjust(self, gradient, learning_rate):
        if self.m is None or self.v is None:
            self.m = np.zeros_like(gradient)
            self.v = np.zeros_like(gradient)

        self.t += 1

        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient

        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(gradient)

        # m_corrected = self.m / (1 - np.power(self.beta1, self.t))
        # v_corrected = self.v / (1 - np.power(self.beta2, self.t))

        step_size = learning_rate * self.m / np.sqrt(self.v + self.epsilon)

        return step_size
