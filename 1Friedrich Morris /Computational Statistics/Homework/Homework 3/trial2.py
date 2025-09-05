# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 14:01:41 2023

@author: 49163
"""

# Trial 2 (Support Vector Machines)

# Data Preprocessing and SVM Model

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# Load the training and test data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Check for missing values and handle them (e.g., impute with median)
train_data.fillna(train_data.median(), inplace=True)
test_data.fillna(test_data.median(), inplace=True)

# Loop through columns to identify and handle non-numeric columns
for column in train_data.columns:
    if train_data[column].dtype == 'object':
        train_data = pd.get_dummies(train_data, columns=[column], drop_first=True)
        test_data = pd.get_dummies(test_data, columns=[column], drop_first=True)

# Normalize or standardize numerical features
scaler = StandardScaler()
num_features = ['rectal_temp', 'pulse', 'respiratory_rate']
train_data[num_features] = scaler.fit_transform(train_data[num_features])
test_data[num_features] = scaler.transform(test_data[num_features])

# Model selection: Support Vector Machines (SVM)

# Features and target variable
X = train_data.drop(['id', 'outcome'], axis=1)
y = train_data['outcome']

# Initialize and train the SVM classifier
svm_classifier = SVC(probability=True, random_state=42)
cross_val_scores = cross_val_score(svm_classifier, X, y, cv=5, scoring='neg_log_loss')

# Print the cross-validation scores (log loss)
print("Cross-validation Log Loss with SVM:", -cross_val_scores.mean())

# Fit the model on the entire training data
svm_classifier.fit(X, y)

# Make predictions on the test data
X_test = test_data.drop(['id'], axis=1)
test_predictions = svm_classifier.predict_proba(X_test)[:, 1]  # Probability of outcome=1

# You can also save these predictions to a CSV file
submission_df = pd.DataFrame({'id': test_data['id'], 'outcome': test_predictions})
submission_df.to_csv('horse_survival_predictions_svm.csv', index=False)

