# -- coding: utf-8 --
"""
Created on Tue Jun 29 20:55:14 2021

equipe: Julane Bezerra
        Rubens Junior
        Aline Soares
        Irlailton Lima
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

parkison = np.genfromtxt('D:/OneDrive/Faculdade/S7/parkinson/parkinson_formated.csv', delimiter = ',')
type(parkison)

features = parkison[:,:754]
features

targets = parkison[:,754]
targets

parameters = {'hidden_layer_sizes':[(3,5),3], 'activation': 
              ['identity', 'logistic', 'tanh', 'relu'], 'max_iter': [200],
              'learning_rate_init': [0.001,0.01,0.1]}
mlp = MLPClassifier(random_state=1, max_iter=300)

clfAccuracy = GridSearchCV(mlp, parameters,cv=5, scoring="accuracy")
clfPrecision = GridSearchCV(mlp, parameters,cv=5, scoring="precision")
clfRecall = GridSearchCV(mlp, parameters,cv=5, scoring="recall")
clfF1 = GridSearchCV(mlp, parameters,cv=5, scoring="f1")

clfAccuracy.fit(features, targets)
clfPrecision.fit(features, targets)
clfRecall.fit(features, targets)
clfF1.fit(features, targets)

sorted(clfAccuracy.cv_results_.keys())
sorted(clfPrecision.cv_results_.keys())
sorted(clfRecall.cv_results_.keys())
sorted(clfF1.cv_results_.keys())

print("Accuracy",clfAccuracy.best_params_)
print("precision",clfPrecision.best_params_)
print("recall",clfRecall.best_params_)
print("f1",clfF1.best_params_)

print(clfPrecision.best_score_)
print(clfAccuracy.best_score_)
print(clfRecall.best_score_)
print(clfF1.best_score_)

print(sum(clfPrecision.best_score_)/5)
