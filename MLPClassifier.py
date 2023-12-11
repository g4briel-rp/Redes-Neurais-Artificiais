import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier

iris_dataset = load_iris()

#print("iris_dataset.data:\n", iris_dataset.data)
#print("iris_dataset.target:\n", iris_dataset.target)
#print("iris_dataset.feature_names:\n", iris_dataset.feature_names)

# cria um dataframe com os dados das amostras (sem a categoria)
data = pd.DataFrame(iris_dataset.data, columns=iris_dataset.feature_names)
# print("data.shape: ", data.shape, "\n", data)

# adiciona a coluna target, com os dados de saída correspondentes
data['target'] = iris_dataset.target
#print("data.shape: ", data.shape, "\n", data)

# seleciona 5 amostras para usar como validação posterior
validation_data = data.sample(10)
# remove as amostras de validação de "data" 
data.drop(validation_data.index, axis=0, inplace=True)

validation_X = validation_data.drop('target', axis=1)
validation_y = validation_data.drop(['sepal length (cm)', 'sepal width (cm)',
                                     'petal length (cm)', 'petal width (cm)'],
                                     axis=1)

# cria uma copia de data, sem a coluna target
features = data.drop('target', axis=1)
#print("features: ", features.shape, "\n", features)

# cria dois conjuntos de treinamento e teste a partir de "data"
# x_train/x_test tem os dados das amostras, y_train/y_test tem as saidas correspondentes
# consulte a função train_test_split para conhecer os parâmetros
x_train, x_test, y_train, y_test = train_test_split(features, data['target'], random_state=42, test_size=0.25)

# MLPClassifier é o perceptron multicamadas do scikit-learn
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
# Consulte a documentação para obter maiores detalhes dos parâmetros de criação
# Aqui o classificador é criado com no máximo 1000 épocas 
# ex: model = MLPClassifier(max_iter=1000)
model = MLPClassifier(hidden_layer_sizes=(5,5,5), max_iter=1000, activation = 'relu',solver='adam')

# executa o processo de treinamento em cima dos dados x_train/y_train
model.fit(x_train, y_train)

# verifica a qualidade do treinamento com os dados de teste
acc = model.score(x_test, y_test)
print("Acuracia do modelo: ", acc)

res = model.predict(validation_X)
print("Predicted: ", res.flatten())
print("Expected: ", validation_y.to_numpy().flatten())
