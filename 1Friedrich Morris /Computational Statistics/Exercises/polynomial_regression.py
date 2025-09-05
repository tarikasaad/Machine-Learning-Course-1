# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 13:40:06 2023

@author: 49163
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def mse(model, X, y):
    return (((y-model.predict(X))**2).mean())

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 2.5])
y = np.array([7.6, 5.0, 4.0, 3.5, 4.0, 5.0, 7.0, 4.0])

plt.scatter(x,y)

polyreg = make_pipeline(PolynomialFeatures(degree = 4), LinearRegression())
# polyreg = LinearRegression()

X = x.reshape(-1, 1)
# X_transformed = np.append(X, X**2, axis = 1)
# X_transformed = np.append(X_transformed, X**3, axis = 1)

polyreg.fit(X, y)

X_seq = np.linspace(0,10,11).reshape(-1,1)
# X_seq_transformed = np.append(X_seq, X_seq**2, axis = 1)
# X_seq_transformed = np.append(X_seq_transformed, X_seq**3, axis = 1)

plt.plot(X_seq, polyreg.predict(X_seq), color="black")
# plt.plot(X_seq, polyreg.predict(X_seq_transformed), color="black")
plt.show()

print(mse(polyreg, X, y))