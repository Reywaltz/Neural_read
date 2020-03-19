import numpy as np
from pprint import pprint
from Network import Neural_neutwork

inp_n = 15
out_n = 10
hid_n = 10
learn_rate = 0.1

n = Neural_neutwork(inp_n, out_n, hid_n, learn_rate)

with open("numbers.csv", 'r') as f:
    data = f.readlines()

data = data[1:]
train_data = data


def preparation(str_matrix):
    f_matrix = []

    targets = []

    for line in str_matrix:
        target = np.zeros(out_n) + 0.1
        target[int(line[0])] = 0.99
        targets.append(target)
        line = line.split(',')
        data = (np.asfarray(line[1:]) / 1.0 * 0.99) + 0.1
        f_matrix.append(data)
    return f_matrix, targets 

matrix, targets = preparation(train_data)
ephos = 500

for e in range(ephos):
    for index, row in enumerate(matrix):
        n.train(row, targets[index])

for i in range(10):
    output = n.query(matrix[i])
    correct_label = np.argmax(targets[i])
    predict_label = np.argmax(output)
    print("want - {}, have - {}".format(str(correct_label), str(predict_label)))


# pprint(n.w_ih)
# pprint(n.w_ho)
# print(np.argmax((n.query([1,1,1,1,0,0,1,1,0,1,0,1,1,1,1]))))