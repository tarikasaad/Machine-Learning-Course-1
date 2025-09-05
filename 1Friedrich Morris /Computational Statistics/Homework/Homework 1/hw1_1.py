# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 15:21:26 2023

@author: 49163
"""
print()
print("author: Robert Smith")
print("Group: Robert, Joaquin, Louis")
print()
print()
print("Homework 1")
print("Part 1")
print()
print()
print()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

df = pd.read_csv("hw1.csv")



# a)

print("a)")
print()
print()
print("Graph plotted")
print()
print()
print()

x = np.array(df['0'])
y = np.array(df['1'])

plt.scatter(x, y)

plt.show()




# b)

print("b)")
print()
print()

# function for mse:
def mse (model, X, y):
    return ((y-model.predict(X))**2).mean()

# create an array of the x variables in a 2nd dimension:
X = np.array(x).reshape(-1,1)

# # creating models (hardcoded):
# model1 = make_pipeline(PolynomialFeatures(degree = 1), LinearRegression())
# model1.fit(X,y)
# model2 = make_pipeline(PolynomialFeatures(degree = 2), LinearRegression())
# model2.fit(X,y)
# model3 = make_pipeline(PolynomialFeatures(degree = 3), LinearRegression())
# model3.fit(X,y)
# model...


# continuation of creating regression models and implementing k-fold valuation::
# Creates 20 models with degrees ranging from 1 to 20
# and stores them in the models dictionary. You can access each model by its name,
# such as models["model1"], models["model5"], and so on, up to models["model20"].

# Define a range of polynomial degrees to test (1 to 20):
degrees = range(1, 21)

# Dictionary to store MSE results:
models = {}

# Create a list to store the models:
models_list = []

# Create an empty list to store MSE values for each degree:
mse_values = []

# Loop through each degree and perform 5-fold cross-validation:
for degree in degrees:
    # Create polynomial features:
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    
    # Perform linear regression:
    model = LinearRegression()
    
    # Calculate MSE using cross-validation:
    mse_score = -cross_val_score(model, X_poly, y, cv=5, scoring='neg_mean_squared_error').mean()
    
    mse_values.append(mse_score)
    
    # creating models:
    model_name = f"model{degree}"
    models_list.append(model_name)
    models[f"model{degree}"] = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    models[f"model{degree}"].fit(X, y)
    
    # Calculate and print MSE for the current model on the data set:
    # mse_train = mse(models[f"model{degree}"], X, y)
    # print(f"MSE for {model_name}: {mse_train}")
    # print()

# Find the degree with the lowest MSE:
best_degree = degrees[np.argmin(mse_values)]
best_mse = np.min(mse_values)

print(f"Best polynomial degree: {best_degree}")
print(f"Validation MSE for best degree: {best_mse}")
print()
print()


# Which model has the lowest MSE?

# According to the the k-fold validation, the model with the lowest MSE, in this scenario
# is model 4 with degree = 4. This is because it is not overly precise to the training data,
# nor is it too vague, such that is is able to more accurate in general, when compared to
# other models for predicting future values. 
print("Which model has the lowest MSE?")
print()
print("According to the the k-fold validation, the model with the lowest MSE, in this scenario is model 4 with degree = 4. This is because it is not overly precise to the training data, nor is it too vague, such that is is able to more accurate in general, when compared to other models for predicting future values.")
print()
print()
print()



# c)

print("c)")
print()
print()

# Set aside 10% of the data in a test set
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1)

# Define a range of polynomial degrees to test (1 to 20):
degrees = range(1, 21)

# Dictionary to store MSE results:
mse_results = {}

# Dictionary to store predictions for each degree
predictions = {}

# Loop through each degree and perform 5-fold cross-validation:
for degree in degrees:
    # Create polynomial features:
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X_train_val)
    
    # Perform linear regression:
    model = LinearRegression()
    
    # Fit the model
    model.fit(X_poly, y_train_val)
    
    # Perform 5-fold cross-validation and get predictions
    y_pred = cross_val_predict(model, X_poly, y_train_val, cv=5)
    
    # Calculate MSE using your custom function
    mse_value = mse(model, X_poly, y_train_val)
    
    # Store MSE in the results dictionary
    mse_results[degree] = mse_value
    
    # Store predictions for this degree
    predictions[degree] = y_pred

# Find the degree with the lowest MSE:
best_degree = min(mse_results, key=mse_results.get)
best_mse = mse_results[best_degree]

# Calculate the expected generalization error on the test set
poly_features = PolynomialFeatures(degree=best_degree)
X_poly_test = poly_features.fit_transform(X_test)
model = LinearRegression()

# Fit the model on the test set
model.fit(X_poly_test, y_test)
y_pred_test = model.predict(X_poly_test)

# Calculate generalization error using your custom function
generalization_error = mse(model, X_poly_test, y_test)

print(f"Best polynomial degree: {best_degree}")
print(f"Test MSE for best degree: {best_mse}")
print(f"Expected generalization error on the test set: {generalization_error}")
print()
print()
print()



# d)

print("d)")
print()
print()

# chose model 13 as after running a test with a 1000 iterattion long loop, model 13 was most commonly the optimal model (had the lowest mse value the most)

# Define the best model based on tests for provided data set with 10% as test set:
best_degree = 13

# Create polynomial features with the best degree
best_poly_features = PolynomialFeatures(degree=best_degree)
X_poly = best_poly_features.fit_transform(X)  # Assuming X contains your entire dataset

# Create and train the best model on the entire dataset
best_model = LinearRegression()
best_model.fit(X_poly, y)  # Assuming y contains the corresponding target values

# Calculate MSE on the entire dataset using the custom mse function
mse_score = mse(best_model, X_poly, y)

# Print a message indicating the best model and its MSE score
print(f"Trained the best model with degree {best_degree} on the entire dataset.")
print(f"MSE Score of the best model: {mse_score}")
print()




