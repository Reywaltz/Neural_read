import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, weigth, bias):
        self.weigth = np.array(weigth)
        self.bias = bias
    
    def feed_Forward(self, x):
        return sigmoid(np.dot(self.weigth, x) + self.bias)

