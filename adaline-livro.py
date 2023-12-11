import numpy as np

# amostras de treinamento
X = np.array([[1.0, 1.0],[1.0,0.0],[0.0,1.0],[0.0,0.0]])
# saidas esperadas
d = np.array([1,-1,-1,-1])
# pesos iniciais
w = np.random.random(X.shape[1])
# bias
b = np.random.random()
# taxa de aprendizagem
n = 0.3
# taxa de parada
stop = 0.001

def eqm(X, d, w, b):
    eqm_val = 0
    for k in range(X.shape[0]):
        u = X[k].dot(w) + b
        erro = (d[k] - u)**2
        eqm_val = eqm_val + erro 
    eqm_val = eqm_val / X.shape[0]
    return eqm_val

def adaline(X, d, w, b, n, stop):
    epocas = 0
    while True:
        eqm_ant = eqm(X, d, w, b)
        for k in range(X.shape[0]):
            u = X[k].dot(w) + b
            erro = d[k] - u
            #atualiza os pesos e bias
            for i in range(w.shape[0]):
                w[i] += n*erro*X[k][i]
            b += n*erro
        eqm_atual = eqm(X, d, w, b)
        if abs(eqm_atual-eqm_ant) <= stop:
            print("Treinamento encerrado com #",epocas)
            return w,b
        epocas += 1

def signal(val):
    if val >= 0:
        return 1
    else:
        return -1

def predict(X, w, b):
    for k in range(X.shape[0]):
        u = X[k].dot(w)+b
        y = signal(u)
        print("Amostra = ", X[k], " E-logico = ", y)

w,b=adaline(X,d,w,b,n,stop)
predict(X,w,b)
