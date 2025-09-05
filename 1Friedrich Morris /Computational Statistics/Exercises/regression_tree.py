#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: witkowski
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

wage = pd.read_csv("Wage.csv")

x = wage["age"]
y = wage["wage"]

X = np.array(x).reshape(-1,1)

X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.1, random_state=10)

model1=make_pipeline(PolynomialFeatures(degree = 1), LinearRegression())
model1.fit(X_train,y_train)
model5=make_pipeline(PolynomialFeatures(degree = 5), LinearRegression())
model5.fit(X_train,y_train)
model12=make_pipeline(PolynomialFeatures(degree = 12), LinearRegression())
model12.fit(X_train,y_train)

tree_regr = DecisionTreeRegressor(min_samples_split = 5)
tree_regr.fit(X_train, y_train)

plt.figure()
plt.scatter(X_train, y_train, color = "black")
plt.scatter(X_validate, y_validate, color = "grey")

X_seq = np.linspace(0,100,1000).reshape(-1,1)
plt.plot(X_seq, model1.predict(X_seq), color="blue")
plt.plot(X_seq, model5.predict(X_seq), color="red")
plt.plot(X_seq, model12.predict(X_seq), color="green")
plt.plot(X_seq, tree_regr.predict(X_seq), color = "yellow")
plt.ylim([0, 400])
plt.show()

def mse(model, X, y):
    return ((y-model.predict(X))**2).mean()

print("MSE of linear function (black):", mse(model1, X_validate, y_validate))
print("MSE of polynomial with degree 5 (red):", mse(model5, X_validate, y_validate))
print("MSE of polynomial with degree 12 (green):", mse(model12, X_validate, y_validate))
print("MSE of decision tree (yellow):", mse(tree_regr, X_validate, y_validate))