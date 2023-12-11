import csv
import random
import numpy as np

def sinal(u):
    if u >= 0:
        return 1
    else:
        return -1

if __name__ == '__main__':
    x = []
    d = []

    with open('dataset-laboratrio-perceptron.csv', mode='r') as arq:
        leitor = csv.reader(arq, delimiter=',')
        linhas = 0
        for coluna in leitor:
            aux = []
            if linhas != 0:
                aux = [float(coluna[1]), float(coluna[2]), float(coluna[3])]
                x.append(aux)
                d.append(int(coluna[4]))
            
            linhas += 1
        print(f'Linhas lidas {linhas}')

    w = np.array([random.random(), random.random(), random.random()])

    tax = 0.05

    epoca = 0

    limiar = random.random()

    print('pesos iniciais:')
    print(w)
    print('limiar inicial:')
    print(limiar)
    print()

    while(True):
        erro = False
        for i in range(len(x)): 
            u = ((w[0]) * float(x[i][0])) + (w[1] * float(x[i][1])) + (w[2] * float(x[i][2])) - limiar
            y = sinal(u)
            if (y != d[i]):
                w[0] = w[0] + (tax * (int(d[i]) - y) * float(x[i][0]))
                w[1] = w[1] + (tax * (int(d[i]) - y) * float(x[i][1]))
                w[2] = w[2] + (tax * (int(d[i]) - y) * float(x[i][2]))
                limiar = limiar + (tax * (int(d[i]) - y) * (-1))
                erro = True
        if erro == False:
            break
        epoca += 1

    print('pesos finais:')
    print(w)
    print('limiar inicial:')
    print(limiar)
    print('epocas: ', epoca)
    print()

    amostras = np.array([[-0.3665, 0.0620, 5.9891], 
                         [-0.7842, 1.1267, 5.5912], 
                         [0.3012, 0.5611, 5.8234], 
                         [0.7757, 1.0648, 8.0677], 
                         [0.1570, 0.8028, 6.3040], 
                         [-0.7014, 1.0316, 3.6005], 
                         [0.3748, 0.1536, 6.1537], 
                         [-0.6920, 0.9404, 4.4058], 
                         [-1.3970, 0.7141, 4.9263], 
                         [-1.8842, -0.2805, 1.2548]])

    for i in range(len(amostras)):
        u = (w[0] * amostras[i][0]) + (w[1] * amostras[i][1]) + (w[2] * amostras[i][2]) - limiar
        y = sinal(u)

        if y == -1:
            print('A amostra ', amostras[i][0], '\t', amostras[i][1], '\t', amostras[i][2], 'eh da classe P1')
        elif y == 1:
            print('A amostra ', amostras[i][0], '\t', amostras[i][1], '\t', amostras[i][2], 'eh da classe P2')