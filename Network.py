import numpy as np
import scipy.special


class Neural_neutwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learn_rate):
        """
        param: input_nodes : число входных узлов
        param: hidden_nodes : число скрытых узлов
        param: output_nodes : число выходных узлов
        param: learn_rate : скорость обучение
        param: w_ih : матрица коэф. весов между входным и скрытым узлами
        param: w_ho : матрица коэф. весов между скрытым и выходным узлами
        """
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learn_rate = learn_rate

        self.w_ih = np.random.normal(0.0, pow(self.hidden_nodes, -0.5),
                                     (self.hidden_nodes, self.input_nodes))

        self.w_ho = np.random.normal(0.0, pow(self.output_nodes, -0.5),
                                     (self.output_nodes, self.hidden_nodes))

        self.activate = lambda x: scipy.special.expit(x)

    def train(self, input_list, targets_list):
        # превращаем входной/целевой вектор в матрицу размерности 2 и транспонируем
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        # создаем скрытый слой путем матричного перемножения весов 
        # матрицы входного-скрытого слоя на входной вектор
        # и прогоняем через функцию активации
        hidden_inputs = np.dot(self.w_ih, inputs)
        hidden_outputs = self.activate(hidden_inputs)

        # создаем выходной слой путем матричного перемножения весов
        # матрицы скрытого-выходного слоя на скрытый слой
        # и прогоняем через функцию активации
        final_inputs = np.dot(self.w_ho, hidden_outputs)
        final_output = self.activate(final_inputs)

        """Вычисление ошибки"""
        output_errors = targets - final_output

        """
        ошибки скрытого слоя - это ошибки output_errors,
        распределенные пропорционально весовым коэффициентам связей
        и рекомбинированные на скрытых узлах
        """
        hidden_errors = np.dot(self.w_ho.T, output_errors)

        """
        Обновление весов между скрытым и выходным
        """
        self.w_ho += self.learn_rate * np.dot((output_errors * final_output *
                                               (1.0 - final_output)),
                                              np.transpose(hidden_outputs))

        """
        Обновление весов между входным и скрытым
        """
        self.w_ih += self.learn_rate * np.dot((hidden_errors * hidden_outputs *
                                               (1.0 - hidden_outputs)),
                                              np.transpose(inputs))

    def query(self, input_list):
        """
        Метод вычисления результата нейронной сети
        @param : input_list : список входных значений
        """
        inputs = np.array(input_list, ndmin=2).T

        hidden_inputs = np.dot(self.w_ih, inputs)
        hidden_output = self.activate(hidden_inputs)

        final_input = np.dot(self.w_ho, hidden_output)
        final_output = self.activate(final_input)

        return final_output
