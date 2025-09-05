#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: witkowski
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import KFold 

def accuracy(model, X, y):
    return np.mean(model.predict(X) == y)

breast_cancer = datasets.load_breast_cancer()
X = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)
y = breast_cancer.target
X = np.array(X)

tree_classifier = DecisionTreeClassifier()
bagged_trees_classifier = RandomForestClassifier(n_estimators=100, max_features=None)
RF_classifier = RandomForestClassifier(n_estimators=100, max_features="sqrt")
RF_classifier2 = RandomForestClassifier(oob_score = True, n_estimators=100, max_features="sqrt")
RF_classifier2.fit(X, y)

k=5
kf = KFold(n_splits=k, shuffle=True)

avg_acc_score = np.zeros(3)
    
for train_indices, validation_indices in kf.split(X):
    X_train, X_validation = X[train_indices,:], X[validation_indices,:]
    y_train, y_validation = y[train_indices], y[validation_indices]

    tree_classifier.fit(X_train,y_train)
    avg_acc_score[0] += accuracy(tree_classifier, X_validation, y_validation)

    bagged_trees_classifier.fit(X_train,y_train)
    avg_acc_score[1] += accuracy(bagged_trees_classifier, X_validation, y_validation)

    RF_classifier.fit(X_train,y_train)
    avg_acc_score[2] += accuracy(RF_classifier, X_validation, y_validation)

avg_acc_score = avg_acc_score/k

print("Tree accuracy (CV)                        :", avg_acc_score[0])
print("Bagged Trees accuracy (CV)                :", avg_acc_score[1])
print("RF accuracy (CV                           :", avg_acc_score[2])
print("RF accuracy (OOB)                         :", RF_classifier2.oob_score_)


feature_importance = RF_classifier2.feature_importances_

best_features = np.argsort(feature_importance)[-2:]
print("\nMost important features of last RF model: ", best_features)

X_reduced = X[:,best_features]
RF_classifier2.fit(X_reduced, y)
print("RF with reduced features accuracy (OOB)   :", RF_classifier2.oob_score_)

grid_size=100
x1_min, x2_min = np.min(X_reduced, axis=0)
x1_max, x2_max = np.max(X_reduced, axis=0)
x1_seq = np.linspace(x1_min, x1_max, grid_size)
x2_seq = np.linspace(x2_min, x2_max, grid_size)

X_seq = np.array([[x1, x2] for x1 in x1_seq for x2 in x2_seq])    
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=y, cmap=plt.cm.Paired)

predicted_labels=RF_classifier2.predict(X_seq)
plt.scatter(X_seq[:,0], X_seq[:,1], s=0.1, c=predicted_labels, cmap=plt.cm.Paired)

plt.show()     