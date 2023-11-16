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

        self.log_var = self.log_var_p.forward(input)
        self.mean = self.mean_p.forward(input)
        self.sample = np.exp(0.5 * self.log_var) * self.epsilon + self.mean

        return self.sample


    # def backward(self, output_gradient, learning_rate: float):
    #     gradLogVar = {}
    #     gradMean = {}
    #     tmp = self.output_size * output_gradient.shape[1]
    #
    #     # KL divergence gradients
    #     gradLogVar["KL"] = (self.std - 1) / (2 * tmp)
    #     gradMean["KL"] = self.mean / tmp
    #
    #     # MSE gradients
    #     gradLogVar["MSE"] = 0.5 * output_gradient* self.epsilon * self.std
    #     gradMean["MSE"] = output_gradient
    #
    #     # backpropagate gradients thorugh self.mean and self.logVar
    #     return self.mean_p.backward(gradMean["KL"] + gradMean["MSE"], learning_rate) + self.log_var_p.backward(
    #         gradLogVar["KL"] + gradLogVar["MSE"], learning_rate)
    #     # return self.mean_p.backward(self.mean + output_gradient*output_gradient , learning_rate) + \
    #     #        self.log_var_p.backward(-self.std - np.exp(self.std)*output_gradient, learning_rate)
    def backward(self, output_gradient, learning_rate):
        mean_gradient = self.mean_p.backward(output_gradient + self.mean, learning_rate)
        log_var_gradient = self.log_var_p.backward(0.5 * output_gradient * self.epsilon * np.exp(self.log_var / 2.) + (np.exp(self.log_var) - 1), learning_rate)

        # self.mean_p.weights_accum += self.mean_p.optimizer.adjust(learning_rate, self.mean)
        # self.log_var_p.weights_accum += self.log_var_p.optimizer.adjust(learning_rate, (np.exp(self.log_var) - 1))

        return mean_gradient + log_var_gradient

    def get_KL(self):
        return -0.5 * np.sum(1 + self.log_var - np.square(self.mean) - np.exp(self.log_var))


    def update(self):
        self.log_var_p.update()
        self.mean_p.update()
