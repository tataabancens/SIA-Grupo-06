import numpy as np
from Sigmoid import SigmoidFunction, SimpleFunction



class MultiLayerPerceptron:
    def __init__(self, perceptron_layers: list[int], input_size: int,sigmoid: SigmoidFunction) -> None:
        self.layers = perceptron_layers
        self.layer_count = len(perceptron_layers)
        self.sigmoid = np.vectorize(sigmoid.eval)
        self.biases = [np.ones(i) for i in perceptron_layers] # biases apply to each perceptron, not the input
        self.weights = [np.ones(i) for i in [input_size] + perceptron_layers]

    def feed_forward(self, a: list[float]):
        a = np.array(a)
        if len(a) != self.weights[0].shape[0]:
            raise ValueError("")
        # check that length of 'a' is same as layers[0] 
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w,a) + b)
        return a

    def compute_cost(self, a: list[float], expected: list[float]):
        if len(a) != self.weights[0].shape[0] or len(a) != len(expected):
            raise ValueError("")
        activations = self.feed_forward(a)
        return 0.5 * np.linalg.norm(activations - expected)**2

        
def main():
    p = MultiLayerPerceptron([4,3,3], 4, SimpleFunction)
    print(p.feed_forward([1,2,3,4]))

    
if __name__ == "__main__":
    main()

    
        

    