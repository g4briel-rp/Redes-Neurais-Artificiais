import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

data = np.array([[-0.6508,0.1097,4.0009], 
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
    
target = np.array([-1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,1,1,1,-1,1,1,1,1,-1,-1,1,-1,1])

feature_names = ['x1', 'x2', 'x3']

data = pd.DataFrame(data, columns=feature_names)
# print("data.shape: ", data.shape, "\n", data)

data['d'] = target
# print("data.shape: ", data.shape, "\n", data)

features = data.drop('d', axis=1)
# print("features: ", features.shape, "\n", features)

x_train, x_test, y_train, y_test = train_test_split(features, data['d'], random_state=42, test_size=0.25)

model = MLPClassifier(hidden_layer_sizes=(5,5,5), max_iter=2000, activation = 'relu',solver='adam')

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print("Acuracia do modelo: ", acc)

test = np.array([[-0.3665, 0.0620, 5.9891], 
                 [-0.7842, 1.1267, 5.5912], 
                 [0.3012, 0.5611, 5.8234], 
                 [0.7757, 1.0648, 8.0677], 
                 [0.1570, 0.8028, 6.3040], 
                 [-0.7014, 1.0316, 3.6005], 
                 [0.3748, 0.1536, 6.1537], 
                 [-0.6920, 0.9404, 4.4058], 
                 [-1.3970, 0.7141, 4.9263], 
                 [-1.8842, -0.2805, 1.2548]])

validation_test = np.array([-1,1,1,1,1,1,-1,1,-1,-1])

test = pd.DataFrame(test, columns=feature_names)
# print("test.shape: ", test.shape, "\n", test)

res = model.predict(test)
print("Predicted: ", res.flatten())
print("Expected: ", validation_test)