#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: witkowski
"""

import numpy as np
import matplotlib.pyplot as plt
#from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree 

tree_classifier = DecisionTreeClassifier(criterion="gini", max_leaf_nodes=3)

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

tree_classifier.fit(X,y)
tree.plot_tree(tree_classifier)
plt.show()

grid_size=100
x1_min, x2_min = np.min(X, axis=0)
x1_max, x2_max = np.max(X, axis=0)
x1_seq = np.linspace(x1_min, x1_max, grid_size)
x2_seq = np.linspace(x2_min, x2_max, grid_size)

X_seq = np.zeros((grid_size**2, 2))

for x1 in range(len(x1_seq)):
    for x2 in range(len(x2_seq)):
        X_seq[grid_size*x1+x2] = x1_seq[x1], x2_seq[x2]

## Alternative for lines 40-44 using comprehension:
# X_seq = np.array([[x1, x2] for x1 in x1_seq for x2 in x2_seq])
        
plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Paired)

predicted_labels=tree_classifier.predict(X_seq)
plt.scatter(X_seq[:,0], X_seq[:,1], s=0.1, c=predicted_labels, cmap=plt.cm.Paired)

## From Class 03
# svc1 = SVC(C = 100, kernel="poly", degree=2)
# svc1.fit(X, y)
# plot_svc(svc1, X, y)

plt.show()     

