from Network import Neural_neutwork
from Neuron import Neuron
import numpy as np

def MSE(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

x = np.array([2, 3])

y_true = np.array([1, 0, 0, 1])
y_pred = np.array([0, 0, 0, 0])
network = Neural_neutwork()
print(network.feed_Forward(x)) 
print(MSE(y_true, y_pred))