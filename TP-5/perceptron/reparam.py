import numpy as np

from perceptron.dense import Dense
from perceptron.layer import Layer
from perceptron.optimizer import Optimizer, GradientDescent


class Reparam(Layer):
    def __init__(self, input_size: int, output_size: int, optimizer: Optimizer = GradientDescent()):
        self.mean_p = Dense(input_size, output_size, optimizer)
        self.log_var_p = Dense(input_size, output_size, optimizer)
        self.output_size = output_size

    def forward(self, input):
        self.epsilon = np.random.standard_normal(size=(self.output_size, input.shape[1]))

        self.std = np.exp(self.log_var_p.forward(input))
        self.mean = self.mean_p.forward(input)
        self.sample = self.std*self.epsilon + self.mean

        return self.sample


    def backward(self, output_gradient, learning_rate: float):
        gradLogVar = {}
        gradMean = {}
        tmp = self.output_size * output_gradient.shape[1]

        # KL divergence gradients
        gradLogVar["KL"] = (self.std - 1) / (2 * tmp)
        gradMean["KL"] = self.mean / tmp

        # MSE gradients
        gradLogVar["MSE"] = 0.5 * output_gradient* self.epsilon * self.std
        gradMean["MSE"] = output_gradient

        # backpropagate gradients thorugh self.mean and self.logVar
        return self.mean_p.backward(gradMean["KL"] + gradMean["MSE"], learning_rate) + self.log_var_p.backward(
            gradLogVar["KL"] + gradLogVar["MSE"], learning_rate)
        # return self.mean_p.backward(self.mean + output_gradient*output_gradient , learning_rate) + \
        #        self.log_var_p.backward(-self.std - np.exp(self.std)*output_gradient, learning_rate)

    def get_KL(self):
        return - np.sum(1 + self.std - np.square(self.mean) - np.exp(self.std))


    def update(self):
        self.log_var_p.update()
        self.mean_p.update()
