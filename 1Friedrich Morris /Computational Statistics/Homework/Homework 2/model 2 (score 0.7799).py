# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 15:45:09 2023

@author: 49163
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# Load the data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Feature Engineering
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch']
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch']

# Handling Missing Values
# You may need to handle missing values here

# Prepare the data
y = train_data["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch", "FamilySize"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean CV accuracy:", cv_scores.mean())

# Make predictions
predictions = model.predict(X_test)

# Create a submission file
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('numero2.csv', index=False)
print("Your submission was successfully saved!")