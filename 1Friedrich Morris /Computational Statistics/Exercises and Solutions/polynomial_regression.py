#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: witkowski
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def mse(model, X, y):
    y_hat = model.predict(X)
    return ((y-y_hat)**2).mean()

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 2.5])
y = np.array([7.6, 5.0, 4.0, 3.5, 4.0, 5.0, 7.0, 4.0])

X = x.reshape(-1, 1)
plt.scatter(X,y, color="black")

polyreg = LinearRegression()
polyreg.fit(X,y)
X_seq = np.linspace(0,10,100).reshape(-1,1)
# plt.plot(X_seq, polyreg.predict(X_seq), color="blue")
print("MSE of linear model:", mse(polyreg, X, y))

X_transformed = np.append(X, X**2, axis = 1)
polyreg_squared = LinearRegression()
polyreg_squared.fit(X_transformed, y)
X_seq_transformed = np.append(X_seq, X_seq**2, axis = 1)
plt.plot(X_seq, polyreg_squared.predict(X_seq_transformed), color="grey")
print("MSE of ... model:", mse(polyreg_squared, X_transformed, y))

polyreg_cubed = LinearRegression()
X_cubed = np.append(X_transformed, X**3, axis = 1)
X_seq_cubed = np.append(X_seq_transformed, X_seq**3, axis = 1)
polyreg_cubed.fit(X_cubed, y)
plt.plot(X_seq, polyreg_cubed.predict(X_seq_cubed), color="darkblue")
print("MSE of ... model:", mse(polyreg_cubed, X_cubed, y))

polyreg_8=make_pipeline(PolynomialFeatures(degree = 8), LinearRegression())
polyreg_8.fit(X, y)
plt.plot(X_seq, polyreg_8.predict(X_seq), color="red")
print("MSE of ... model:", mse(polyreg_8, X, y))

plt.ylim([0,10])

plt.show()