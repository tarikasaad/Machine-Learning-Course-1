#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: witkowski
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

def plot_svc(svc, X, y, h=0.02, pad=0.25):
    x_min, x_max = X[:, 0].min()-pad, X[:, 0].max()+pad
    y_min, y_max = X[:, 1].min()-pad, X[:, 1].max()+pad
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.2)

    plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=plt.cm.Paired)
    # Support vectors indicated in plot by vertical lines
    sv = svc.support_vectors_
    plt.scatter(sv[:,0], sv[:,1], c='k', marker='|', s=100, linewidths=1)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    print('Number of support vectors: ', svc.support_.size)

np.random.seed(10)
cluster_size = 10

mean1 = np.array([1,1])
cov1 = np.eye(2)

X_cluster1 = np.random.multivariate_normal(mean1, cov1, cluster_size)

mean2 = np.array([2,2])
cov2 = np.eye(2)

X_cluster2 = np.random.multivariate_normal(mean2, cov1, cluster_size)

X = np.append(X_cluster1, X_cluster2, axis=0)

y = np.append(np.full(10,1), np.full(10,-1), axis=0)

plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=plt.cm.Paired)

svc1 = SVC(C = 1, kernel = "rbf", degree = 1)
svc1.fit(X,y)
plot_svc(svc1, X, y)
