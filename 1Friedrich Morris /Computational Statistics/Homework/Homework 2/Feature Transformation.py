# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 20:39:03 2023

@author: 49163
"""



import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Load data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Define a list of feature transformation options
feature_options = [
    ("FamilySize", ["Pclass", "Sex", "SibSp", "Parch"]),
    ("NoFamily", ["Pclass", "Sex"]),
    ("PclassOnly", ["Pclass", "Sex"]),
    # Add more options as needed
]

# Initialize variables to keep track of the best model and its score
best_model = None
best_score = 0

# Iterate through feature transformation options
for option_name, feature_list in feature_options:
    # Feature Engineering
    if "FamilySize" in feature_list:
        train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch']
        test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch']

    # Data Preparation
    y = train_data["Survived"]  # Target variable
    X = train_data[feature_list]  # Select relevant features
    X_test = test_data[feature_list]  # Preprocess test data similarly

    # Create a column transformer to one-hot encode categorical features
    categorical_features = ["Pclass", "Sex"]
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)
        ])

    # Append a classifier to the preprocessor
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=1
    )

    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', clf)])

    model.fit(X, y)

    # Cross-validation to evaluate model
    cv_scores = cross_val_score(model, X, y, cv=5)
    avg_cv_score = cv_scores.mean()

    # Check if this model has a better score than the previous best model
    if avg_cv_score > best_score:
        best_score = avg_cv_score
        best_model = model
        best_feature_option = option_name

# Make predictions with the best model
predictions = best_model.predict(X_test)

# Create submission file
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv(f'bestmodel_with_{best_feature_option}_transform.csv', index=False)
print(f"Your submission with {best_feature_option} feature transformation was successfully saved as 'bestmodel_with_{best_feature_option}_transform.csv!'")
