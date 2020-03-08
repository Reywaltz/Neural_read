import numpy as np
from Neuron import Neuron

class Neural_neutwork:
    def __init__(self):
        weigth = np.array([0, 1])
        bias = 0

        self.h1 = Neuron(weigth, bias)
        self.h2 = Neuron(weigth, bias)
        self.o1 = Neuron(weigth, bias)

    def feed_Forward(self, x):
        "Результаты узлов 1 и 2"
        out_h1 = self.h1.feed_Forward(x)
        out_h2 = self.h2.feed_Forward(x)

        "Выходы из предыдущих узлов будут входными параметрами для следующего узла"
        return self.o1.feed_Forward(np.array([out_h1, out_h2]))