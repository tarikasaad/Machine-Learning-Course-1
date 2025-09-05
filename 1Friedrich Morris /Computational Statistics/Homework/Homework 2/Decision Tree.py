# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 19:22:47 2023

@author: 49163
"""

# Decision Tree Classifier (score on kaggle = 0.72248)

# import data:

import pandas as pd

train_data = pd.read_csv("train.csv")
train_data.head()

test_data = pd.read_csv("test.csv")
test_data.head()

# Data Cleaning: Fill missing age values with the mean age of all passengers
mean_age = train_data["Age"].mean()
train_data["Age"].fillna(mean_age, inplace=True)
test_data["Age"].fillna(mean_age, inplace=True)

# create model to predict if passenger survives:

from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer

y = train_data["Survived"]

features = ["Pclass", "Sex", "Age", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])

# Use SimpleImputer to handle missing values in the test set as well
X_test = pd.get_dummies(test_data[features])
imputer = SimpleImputer(strategy="mean")
X_test = pd.DataFrame(imputer.fit_transform(X_test), columns=X_test.columns)

model = DecisionTreeClassifier(random_state=1)
model.fit(X, y)

# Predict survival values:
predictions = model.predict(X_test)

# Output results:
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('Tree_submission.csv', index=False)
print("Your submission was successfully saved!")
