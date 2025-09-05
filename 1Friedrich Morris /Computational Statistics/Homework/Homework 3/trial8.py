# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 15:08:09 2023

@author: 49163
"""

# Trial 8 (Random Forest with more features)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Load the data (change file paths as needed)
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Selected numerical features
selected_features = [
    'rectal_temp',
    'packed_cell_volume',
    'total_protein',
]

# Drop all other columns except 'id' for test_data
train_data = train_data[selected_features + ['outcome']]  # Include the outcome column
test_data = test_data[['id'] + selected_features]

# Fill missing values with column medians
train_data.fillna(train_data.median(), inplace=True)
test_data.fillna(test_data.median(), inplace=True)

# Separate features (X) and target (y) for training
X = train_data.drop('outcome', axis=1)
y = train_data['outcome']

# Initialize and train a Random Forest classifier
random_forest = RandomForestClassifier()
cross_val_scores = cross_val_score(random_forest, X, y, cv=5, scoring='neg_log_loss')

# Evaluate the model using cross-validation
mean_log_loss = -cross_val_scores.mean()
print("Mean Log Loss with Random Forest:", mean_log_loss)

# Train the model on the full training data
random_forest.fit(X, y)

# Make predictions on the test data
test_predictions = random_forest.predict_proba(test_data[selected_features])

# Assuming you have 'id' column in your test_data, create a submission DataFrame
submission_df = pd.DataFrame({
    'id': test_data['id'],
    'outcome': test_predictions[:, 1]  # Replace with the appropriate index (0, 1, or 2)
})

# Save the submission DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False)
