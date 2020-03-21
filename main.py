import numpy as np
from Network import Neural_neutwork

inp_n = 36
out_n = 26
hid_n = 26
learn_rate = 0.1
n = Neural_neutwork(inp_n, out_n, hid_n, learn_rate)

with open("letters.csv", 'r') as f:
    data = f.readlines()

letters = data[0].strip()
train_data = data[1:]
line = letters.split(',')

dct = {i: x for i, x in enumerate(line)}


def preparation(str_matrix):
    f_matrix = []
    targets = []

    for line in str_matrix:
        line = line.strip().split(',')
        data = (np.asfarray(line[1:]) / 1.0 * 0.99) + 0.1
        f_matrix.append(data)
        target = np.zeros(out_n) + 0.1
        target[int(line[0])] = 0.99
        targets.append(target)

    return f_matrix, targets


matrix, targets = preparation(train_data)
ephos = 500

for e in range(ephos):
    for index, row in enumerate(matrix):
        n.train(row, targets[index])

for i in range(hid_n):
    output = n.query(matrix[i])
    correct_label = np.argmax(targets[i])
    predict_label = np.argmax(output)
    print("Прогноз - {}, Результат - {}".format(str(dct[correct_label]),
                                                str(dct[predict_label])))

