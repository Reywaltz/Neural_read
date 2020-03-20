num = 9
letters = "A,B,C,D,E,F,G,H,I"
line = letters.split(',')


dct = {x: x+1 for x in range(num)}
for i in range(len(line)):
    dct[i] = line[i]
print(dct)
