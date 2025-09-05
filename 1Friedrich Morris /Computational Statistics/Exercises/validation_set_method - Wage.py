# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 16:04:23 2023

@author: 49163
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

def mse(model, X, y):
    return (((y-model.predict(X))**2).mean())

wage = pd.read_csv("Wage.csv")

x = np.array(wage["age"])
y = np.array(wage["wage"])

X = x.reshape(-1,1)

X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size = 0.5, #random_state = 10)
                                                                                    )
plt.scatter(X_train, y_train, color = "black")

plt.scatter(X_validate, y_validate, color = "gray")


model1 = make_pipeline(PolynomialFeatures(degree = 1), LinearRegression())
model1.fit(X_train, y_train)

model5 = make_pipeline(PolynomialFeatures(degree = 5), LinearRegression())
model5.fit(X_train, y_train)

model12 = make_pipeline(PolynomialFeatures(degree = 12), LinearRegression())
model12.fit(X_train, y_train)

X_seq = np.linspace(10, 85,81).reshape(-1,1)

plt.plot(X_seq, model1.predict(X_seq), color = "blue")
plt.plot(X_seq, model5.predict(X_seq), color = "red")
plt.plot(X_seq, model12.predict(X_seq), color = "green")
plt.ylim((-100, 400))


print(mse(model1, X_train, y_train))
print(mse(model5, X_train, y_train))
print(mse(model12, X_train, y_train))
print()
print(mse(model1, X_validate, y_validate))
print(mse(model5, X_validate, y_validate))
print(mse(model12, X_validate, y_validate))