import random
import numpy as np

def sinal(u):
    if u >= 0:
        return 1
    else:
        return -1

if __name__ == '__main__':
    x = np.array([[-0.6508,0.1097,4.0009], 
                  [-1.4492,0.8896,4.4005], 
                  [2.085,0.6876,12.071], 
                  [0.2626,1.1476,7.7985], 
                  [0.6418,1.0234,7.0427], 
                  [0.2569,0.673,8.3265], 
                  [1.1155,0.6043,7.4446], 
                  [0.0914,0.3399,7.0677], 
                  [0.0121,0.5256,4.6316], 
                  [-0.0429,0.466,5.4323], 
                  [0.434,0.687,8.2287], 
                  [0.2735,1.0287,7.1934], 
                  [0.4839,0.4851,7.485], 
                  [0.4089,-0.1267,5.5019], 
                  [1.4391,0.1614,8.5843], 
                  [-0.9115,-0.1973,2.1962], 
                  [0.3654,1.0475,7.4858], 
                  [0.2144,0.7515,7.1699], 
                  [0.2013,1.0014,6.5489], 
                  [0.6483,0.2183,5.8991], 
                  [-0.1147,0.2242,7.2435], 
                  [-0.797,0.8795,3.8762], 
                  [-1.0625,0.6366,2.4707], 
                  [0.5307,0.1285,5.6883], 
                  [-1.22,0.7777,1.7252], 
                  [0.3957,0.1076,5.6623], 
                  [-0.1013,0.5989,7.1812], 
                  [2.4482,0.9455,11.2095], 
                  [2.0149,0.6192,10.9263], 
                  [0.2012,0.2611,5.4631]])
    
    d = np.array([-1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,1,1,1,-1,1,1,1,1,-1,-1,1,-1,1])

    w = np.array([random.random(), random.random(), random.random()])

    tax = 0.7

    epoca = 0

    limiar = random.random()

    # print(len(x))
    # print(len(d))

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
            print('A amostra ', x[i][0], ' - ', x[i][1], ' - ', x[i][2], 'eh da classe P1')
        elif y == 1:
            print('A amostra ', x[i][0], ' - ', x[i][1], ' - ', x[i][2], 'eh da classe P2')