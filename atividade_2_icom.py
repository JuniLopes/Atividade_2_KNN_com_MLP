# -- coding: utf-8 --
"""
Created on Tue Jun 29 20:55:14 2021

equipe: Julane Bezerra
        Rubens Junior
        Aline Soares
        Irlailton Lima
"""

import numpy as np
parkinson = np.genfromtxt('D:/OneDrive/Faculdade/S7/parkinson/parkinson_formated.csv', delimiter= ',')
data = parkinson
type(parkinson)

features = data[:,:754]
features
targets = data[:, 754]
targets

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
features_norm = min_max_scaler.fit_transform(features)
features_norm 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features_norm, targets, test_size=0.2)
y_train

print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)

X_train
y_train

#Importa a classe
from sklearn.neighbors import KNeighborsClassifier
#instancia um objeto da tecnica de classificacao KNN
knn = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
# treina um modelo de classificacao
knn.fit(X_train, y_train)
# testa o modelo de classificacao
y_pred=knn.predict(X_test)
print("Classe predita:   ")
print(y_pred)
print("Classe verdadeira:")
print(y_test)