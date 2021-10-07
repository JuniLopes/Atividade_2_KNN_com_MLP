# -- coding: utf-8 --
"""
Created on Tue Jun 29 20:55:14 2021

equipe: Julane Bezerra
        Rubens Junior
        Aline Soares
        Irlailton Lima
"""

import numpy as np

parkison = np.genfromtxt('D:/OneDrive/Faculdade/S7/parkinson/parkinson_formated.csv', delimiter = ',')
type(parkison)

features = parkison[:,:754]
features

targets = parkison[:,754]
targets

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
features_norm = min_max_scaler.fit_transform(features)

from sklearn.model_selection import train_test_split


# divide os dados em dois conjuntos (treino e teste)
X_train, X_test, y_train, y_test = train_test_split(features_norm, targets, test_size=0.2)

from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

mlp=MLPClassifier(activation="tanh",max_iter=600,solver="adam",random_state=2,early_stopping=False,learning_rate_init=0.001)
mlp.fit(X_train,y_train)

plt.plot(mlp.loss_curve_,label="treino")

plt.title("Curva do erro de treinamento \n learning_rate 0.001 \n max_iter 100")
plt.xlabel("Iterações")
plt.ylabel("Erro")
plt.legend()
print (mlp.loss_curve_)