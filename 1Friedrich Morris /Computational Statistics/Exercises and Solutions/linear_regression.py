#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 19:19:31 2022

@author: witkowski
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 2.5])
y = np.array([7.6, 5.0, 4.0, 3.5, 4.0, 5.0, 7.0, 4.0])

plt.scatter(x,y)

polyreg = LinearRegression()

X = x.reshape(-1, 1)

polyreg.fit(X,y)

new_X = np.array([[10]])
polyreg.predict(new_X)

new_X = np.array([[-10],[15],[50]])
polyreg.predict(new_X)

X_seq = np.linspace(0,10,11).reshape(-1,1)
polyreg.predict(X_seq)

plt.plot(X_seq, polyreg.predict(X_seq), color="black")

plt.show()
