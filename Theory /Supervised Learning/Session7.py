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

tree_classfier = DecisionTreeClassifier()
RF_classifier = RandomForestClassifier(oob_score=True) #oob_scores can only be used for all the bagging models
bagged_tree_classifier = RandomForestClassifier(max_features=None)



k = 5

kf = KFold(n_splits=k, shuffle=True)

dt_scores = []
rf_scores = []
bt_scores = []

for train_indices, validate_indices in kf.split(X):
    X_train, X_validate = X.iloc[train_indices, :], X.iloc[validate_indices, :]
    y_train, y_validate = y[train_indices], y[validate_indices]

    tree_classfier.fit(X_train, y_train)
    dt_scores.append(accuracy(tree_classfier, X_validate, y_validate))

    RF_classifier.fit(X_train, y_train)
    rf_scores.append(accuracy(RF_classifier, X_validate, y_validate))

    bagged_tree_classifier.fit(X_train, y_train)
    bt_scores.append(accuracy(bagged_tree_classifier, X_validate, y_validate))

print(f"Decision Tree average accuracy: {np.mean(dt_scores):4f}")
print(f"Random Forest average accuracy: {np.mean(rf_scores):4f}")
print(f"Bagged Tree accuracy: {np.mean(bt_scores):4f}")