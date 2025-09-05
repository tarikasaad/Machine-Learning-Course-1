# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 14:28:14 2023

@author: 49163
"""

# Trial 7 (Decision Tree)

# Data Preprocessing and Decision Tree Model

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Load the training and test data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Check for missing values and handle them (e.g., impute with median)
train_data.fillna(train_data.median(), inplace=True)
test_data.fillna(test_data.median(), inplace=True)

# Model selection: Decision Tree

# Features and target variable
X = train_data.drop(['id', 'outcome'], axis=1)
y = train_data['outcome']

# Initialize and train the Decision Tree classifier
decision_tree_classifier = DecisionTreeClassifier()
cross_val_scores = cross_val_score(decision_tree_classifier, X, y, cv=5, scoring='neg_log_loss')

# Print the cross-validation scores (log loss)
print("Cross-validation Log Loss with Decision Tree:", -cross_val_scores.mean())

# Fit the model on the entire training data
decision_tree_classifier.fit(X, y)

# Make predictions on the test data
X_test = test_data.drop(['id'], axis=1)
test_predictions = decision_tree_classifier.predict(X_test)

# Create a DataFrame for submission
submission_df = pd.DataFrame({'id': test_data['id'], 'outcome': test_predictions})

# Save the predictions to a CSV file
submission_df.to_csv('horse_survival_predictions_decision_tree.csv', index=False)

