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
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
import pandas as pd
from sklearn.model_selection import KFold

def accuracy(model, X, y):
    return np.mean(y_validate == model.predict(X_validate))

breast_cancer = datasets.load_breast_cancer()

X = pd.DataFrame(data=breast_cancer.data, columns = breast_cancer.feature_names)
y = breast_cancer.target

tree_classifier = DecisionTreeClassifier()
bagged_tree_classifier = RandomForestClassifier(n_estimators = 100, max_features = None)
RF_classifier = RandomForestClassifier(n_estimators = 100)
RF_classifier2 = RandomForestClassifier(n_estimators = 100, oob_score=True)

RF_classifier2.fit(X, y)

avg_acc_score = np.zeros(3)

k = 5
kf = KFold(n_splits=k, shuffle=True)

for train_indices, validate_indices in kf.split(X):
    X_train, X_validate = X.iloc[train_indices,:], X.iloc[validate_indices,:]
    y_train, y_validate = y[train_indices], y[validate_indices]
    
    tree_classifier.fit(X_train, y_train)
    bagged_tree_classifier.fit(X_train, y_train)
    RF_classifier.fit(X_train, y_train)
    
    avg_acc_score[0] += accuracy(tree_classifier, X_validate, y_validate)
    avg_acc_score[1] += accuracy(bagged_tree_classifier, X_validate, y_validate)
    avg_acc_score[2] += accuracy(RF_classifier, X_validate, y_validate)

avg_acc_score = avg_acc_score / k    

print("Tree:               ", avg_acc_score[0])
print("Bagged trees:       ", avg_acc_score[1])
print("Random Forest (CV): ", avg_acc_score[2])
print("Random Forest (OOB):", RF_classifier2.oob_score_)

best_features = np.argsort(RF_classifier2.feature_importances_)[-2:]



