#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: witkowski
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

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
cov1 = 1*np.eye(2)
X_cluster1 = np.random.multivariate_normal(mean1, cov1, cluster_size)

mean2 = np.array([2,2])
cov2 = 1*np.eye(2)
X_cluster2 = np.random.multivariate_normal(mean2, cov2, cluster_size)

X = np.append(X_cluster1, X_cluster2, axis=0)

y = np.repeat([1,-1], cluster_size)



# svc1 = SVC(C = 1, kernel="linear")
#svc1 = SVC(C = 100, kernel="poly", degree=3)
# svc1.fit(X, y)


tree_classifier = DecisionTreeClassifier(max_leaf_nodes=7, criterion="entropy")
tree_classifier.fit(X,y)

# tree.plot_tree(tree_classifier)

grid_size = 100

x1_min, x2_min = X.min(axis=0)
x1_max, x2_max = X.max(axis=0)

X_seq = np.zeros((grid_size**2, 2))

x1_seq = np.linspace(x1_min, x1_max, grid_size)
x2_seq = np.linspace(x2_min, x2_max, grid_size)

for x1 in range(grid_size):
    for x2 in range(grid_size):
        X_seq[x1*grid_size+x2] = x1_seq[x1], x2_seq[x2]
        
predicted_labels = tree_classifier.predict(X_seq)

plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Paired) # plotting data points

plt.scatter(X_seq[:,0], X_seq[:,1], s=0.1, c=predicted_labels, cmap=plt.cm.Paired) # plotting grid
    

plt.show()
# plot_svc(svc1, X, y)
