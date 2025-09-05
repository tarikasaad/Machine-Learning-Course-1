# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 17:48:38 2023

@author: 49163
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

def mse(model, X, y):
    return (((y-model.predict(X))**2).mean())

x = np.array([1,1,3,3])
y = np.array([1,2,2,3])

X = x.reshape(-1,1)

model1 = make_pipeline(PolynomialFeatures(degree = 1), LinearRegression())
model1.fit(X, y)

plt.scatter(x,y)

X_seq = np.linspace(0, 4, 11).reshape(-1,1)

plt.plot(X_seq, model1.predict(X_seq), color = "blue")
plt.ylim(0, 4)
plt.xlim(0, 4)

print(mse(model1, X, y))

