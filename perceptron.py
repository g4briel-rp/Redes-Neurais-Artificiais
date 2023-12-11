import random
import numpy as np

def sinal(u):
    if u >= 0:
        return 1
    else:
        return 0

if __name__ == '__main__':
    x = np.array([[0,0], [0,1], [1,0], [1,1]])
    
    d = np.array([0, 0, 0, 1])

    w = np.array([random.random(), random.random()])

    tax = 0.7

    epoca = 0

    limiar = random.random()

    while(True):
        erro = False
        for i in range(len(x)): 
            u = (w[0] * x[i][0]) + (w[1] * x[i][1]) - limiar
            y = sinal(u)
            if (y != d[i]):
                w[0] = w[0] + (tax * (d[i] - y) * x[i][0])
                w[1] = w[1] + (tax * (d[i] - y) * x[i][1])
                limiar = limiar + (tax * (d[i] - y) * (-1))
                erro = True
        if erro == False:
            break
        epoca += 1

    print(w)

    for i in range(len(x)):
        u = (w[0] * x[i][0]) + (w[1] * x[i][1]) - limiar
        y = sinal(u)

        if y == 0:
            print('A amostra ', x[i][0], ' - ', x[i][1], 'eh ', y)
        elif y == 1:
            print('A amostra ', x[i][0], ' - ', x[i][1], 'eh ', y)
