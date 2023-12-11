import numpy as np

# amostras
#              x0,x1
X = np.array([[-1.0,-1.0],
              [-1.0,1.0],
              [1.0,-1.0],
              [1.0,1.0]])


# saida desejada
d = np.array([-1.0,1.0,1.0,1.0])

# taxa de aprendizagem
n = 0.2

bias = np.random.random() 


# params: X = amostras, d = saidas, n = taxa de aprendizagem, b = bias
def adaline(X, d, n, b, stop):
    w = np.random.random(X.shape[1])
    #w = np.array([0.5,0.5])
    epoca = 1 
    h_eqm = []
    #for i in range(epoca):
    while True:
        print("Epoca=", epoca)
        soma_erros = 0.0

        for k in range(X.shape[0]):
            dk = d[k] # saida esperada

            # faz a soma ponderada
            u=0.0
            for j in range(X.shape[1]):
               u += X[k][j] * w[j]
            u += b

            # calcula erro da amostra e soma o erro quadratico
            erro = dk - u
            print("[dk=", dk, " u=", u,"][Erro: ", erro, "]")
            soma_erros += erro**2

            # atualiza os pesos e bias
            for j in range(w.shape[0]):
                w[j] += n * erro * X[k][j]
            b += n * erro
        eqm = soma_erros / X.shape[0]
        print("Erro quadratico medio = ", eqm)
        h_eqm.append(eqm)
        if len(h_eqm)>1 and abs(h_eqm[-1]-h_eqm[-2]) <= stop:
            print("Diferenca de eqm = ", abs(h_eqm[-1]-h_eqm[-2]))
            return w,b
        epoca += 1

def signal(u):
    if u >= 0:
        return 1
    else:
        return -1

def predict(X, w, b):
    for k in range(X.shape[0]):
        u = X[k].dot(w)+b
        print("U=", u)
        if signal(u) > 0:
            print("X=", X[k], " or= true ");
        else:
            print("X=", X[k], " or= false");

w,bias = adaline(X, d, n, bias, 0.001)

predict(X, w, bias)

