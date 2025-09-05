# -*- coding: utf-8 -*-
"""
Group Name: Binary Brothers
Group Memebers: Joaquin Rodriguez, Robert Smith, Louis Schmidt
"""


# Code 1 (Random Forest) (score on kaggle based on public dataset = 0.52956)

# Data Preprocessing and Random Forest Model

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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

# Model selection: Random Forest

# Features and target variable
X = train_data.drop(['id', 'outcome'], axis=1)
y = train_data['outcome']

# Initialize and train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
cross_val_scores = cross_val_score(rf_classifier, X, y, cv=5, scoring='neg_log_loss')

# Print the cross-validation scores (log loss)
print("Cross-validation Log Loss:", -cross_val_scores.mean())

# Fit the model on the entire training data
rf_classifier.fit(X, y)

# Make predictions on the test data
X_test = test_data.drop(['id'], axis=1)
test_predictions = rf_classifier.predict_proba(X_test)[:, 1]  # Probability of outcome=1

# You can also save these predictions to a CSV file
submission_df = pd.DataFrame({'id': test_data['id'], 'outcome': test_predictions})
submission_df.to_csv('hw3_code1.csv', index=False)




# Code 2 (Random Forest) (score on kaggle based on public dataset = 0.52978)

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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
num_features = ['rectal_temp', 'pulse', 'respiratory_rate', 'packed_cell_volume', 'total_protein']
train_data[num_features] = scaler.fit_transform(train_data[num_features])
test_data[num_features] = scaler.transform(test_data[num_features])

# Model selection: Random Forest

# Features and target variable
X = train_data.drop(['id', 'outcome'], axis=1)
y = train_data['outcome']

# Initialize and train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
cross_val_scores = cross_val_score(rf_classifier, X, y, cv=5, scoring='neg_log_loss')

# Print the cross-validation scores (log loss)
print("Cross-validation Log Loss:", -cross_val_scores.mean())

# Fit the model on the entire training data
rf_classifier.fit(X, y)

# Make predictions on the test data
X_test = test_data.drop(['id'], axis=1)
test_predictions = rf_classifier.predict_proba(X_test)[:, 1]  # Probability of outcome=1

# You can also save these predictions to a CSV file
submission_df = pd.DataFrame({'id': test_data['id'], 'outcome': test_predictions})
submission_df.to_csv('hw3_code2.csv', index=False)


# Report:
# =============================================================================
# 
# In Code 1, we constructed a Random Forest model for horse outcome prediction. Our journey
# can be broken down into several steps:
# 
# Data Loading: We initiated the process by loading both training and test datasets.
# 
# Handling Missing Data: To maintain data integrity, we addressed missing values in both
# datasets by substituting them with the respective column's median values. We used median
# because is easy to implement and it helps to mantain the distribution of the data
# 
# Categorical Feature Encoding: Since the datasets contained categorical features, we made
# them machine-learning-friendly by employing one-hot encoding. This transformation involved
# creating binary columns for each category, streamlining them for machine learning. We opted
# to retain only one binary column per category to prevent multicollinearity.
# 
# Feature Scaling: We standardized numerical features using the StandardScaler. Standardization
# ensured that these features had a mean of 0 and a standard deviation of 1, making them
# compatible with machine learning models.
# 
# Model Selection: Random Forest: We chose the Random Forest model due to its ensemble
# learning approach, involving multiple decision trees for predictions. We initialized it with
# 100 decision trees.
# 
# Cross-Validation: To assess the model's performance, we utilized 5-fold cross-validation. We
# particularly focused on the log loss metric to evaluate generalization ability and mitigate
# the risk of overfitting.
# 
# Performance Evaluation: We evaluated our model by computing negative log loss scores from
# cross-validation and printing the mean log loss score. This score provided insights into how
# well the model predicted horse outcomes.
# 
# Moving on to Code 2, we adopted the following approach:
# 
# Import Libraries: We began by importing essential libraries. These included pandas for data
# manipulation, StandardScaler for feature scaling, RandomForestClassifier for classification,
# and cross_val_score for cross-validation.
# 
# Load Data: We imported the training and test datasets from CSV files, storing them as 'train_data'
# and 'test_data,' respectively.
# 
# Handle Missing Values: To ensure data completeness, we filled in missing values in both datasets
# with the respective column's median value.
# 
# Data Preprocessing: Data preprocessing was a multifaceted process. It encompassed handling
# non-numeric (categorical) columns, scaling numerical features, and preparing the dataset for
# machine learning. Our steps included one-hot encoding for categorical columns, scaling numeric
# features, segmenting the data into features and the target variable, and excluding 'id' and
# 'outcome' columns from the features.
# 
# Model Selection: Our model of choice was the Random Forest classifier (rf_classifier), initialized
# with 100 estimators.
# 
# Print Cross-Validation Scores: We executed a 5-fold cross-validation with the assistance of
# cross_val_score, calculating the negative log loss for each fold. The results were recorded in
# 'cross_val_scores,' and we showcased the mean log loss score to evaluate our model's performance.
# 
# Train the Model: The Random Forest classifier was trained on the complete training dataset.
# 
# Make Predictions: We prepared the test data by eliminating the 'id' column. The trained model was 
# then employed to predict the probability of the outcome being '1' for each test sample. We stored
# these outcomes in 'test_predictions.'
# 
# Save Predictions: The final step involved organizing the predicted outcomes and corresponding IDs
# in a DataFrame ('submission_df'), which was saved as 'hw3_code2.csv.' Importantly, we excluded the
# index when saving.
# 
# 
# Additional Trials and Model Exploration:
# 
# In both trials, our choice of model was Random Forest, however we were also experimenting with
# different models and features to identify the most effective approach for predicting horse health
# outcomes. We tried different models such as Support Vector Classifier (SVC), polynomial degrees,
# Random Forests, bagged forests, decision trees, and we wanted to include additional features such
# as Temperature deviation of the horses, the hearth rate ratio of the horses and their quantity of
# lesions. 
# 
# However, these model explorations consistently resulted in increased log loss values, indicating a
# decline in prediction performance compared to the baseline Random Forest model. Additionally, the
# previous mentioned features, that we wanted to add, resulted in values error because we could not
# appropriately transform some string values into a numeric format. We tried to use some techniques
# for encoding categorical data but we had no succes.  Despite the extensive experimentation, the Random
# Forest model used in "Code 1" remained the most effective choice for predicting horse outcomes. This
# shows that is is very important to select an appropriate model and feature preprocessing methods when
# tackling classification tasks, as some perform considerably worse than others.
# 
# =============================================================================
