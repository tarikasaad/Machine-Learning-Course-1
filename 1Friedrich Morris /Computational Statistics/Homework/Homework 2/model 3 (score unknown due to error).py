# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 16:19:03 2023

@author: 49163
"""

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

# Load the training and test data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Prepare the data for training
y = train_data["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

# Split the training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

# Create a dictionary to store MSE results and predictions
results = {}

# Polynomial Regression (Degree 2)
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_val = poly.transform(X_val)
poly_reg = LinearRegression()
poly_reg.fit(X_poly_train, y_train)
poly_predictions = poly_reg.predict(X_poly_val)
poly_mse = mean_squared_error(y_val, poly_predictions)
results["Polynomial Regression (Degree 2)"] = {
    "mse": poly_mse,
    "predictions": np.round(poly_reg.predict(poly.transform(X_test)))
}

# Polynomial Regression (Degree 3)
poly = PolynomialFeatures(degree=3)
X_poly_train = poly.fit_transform(X_train)
X_poly_val = poly.transform(X_val)
poly_reg = LinearRegression()
poly_reg.fit(X_poly_train, y_train)
poly_predictions = poly_reg.predict(X_poly_val)
poly_mse = mean_squared_error(y_val, poly_predictions)
results["Polynomial Regression (Degree 3)"] = {
    "mse": poly_mse,
    "predictions": np.round(poly_reg.predict(poly.transform(X_test)))
}

# Regression Tree (Depth 5)
tree_reg = DecisionTreeRegressor(max_depth=5)
tree_reg.fit(X_train, y_train)
tree_predictions = tree_reg.predict(X_val)
tree_mse = mean_squared_error(y_val, tree_predictions)
results["Regression Tree (Depth 5)"] = {
    "mse": tree_mse,
    "predictions": np.round(tree_reg.predict(X_test))
}

# Print the MSE results for all configurations
print("\nMSE Results:")
for model_name, data in results.items():
    print(f"{model_name} - MSE: {data['mse']}")

# Save predictions to CSV for each model
for model_name, data in results.items():
    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': data['predictions']})
    output.to_csv(f'{model_name}_submission.csv', index=False)
    print(f"Predictions for {model_name} were successfully saved to {model_name}_submission.csv")

