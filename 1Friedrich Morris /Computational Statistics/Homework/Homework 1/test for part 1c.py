# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 19:20:09 2023

@author: 49163
"""



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


x = np.array(df['0'])
y = np.array(df['1'])

plt.scatter(x, y)

plt.show()
print()



# b)


# function for mse:
def mse (model, X, y):
    return ((y-model.predict(X))**2).mean()

# create an array of the x variables in a 2nd dimension:
X = np.array(x).reshape(-1,1)






# c)

optimal_list = []

for z in range(0,1000):
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
    
    optimal_list.append(best_degree)
    
print(np.mean(optimal_list))

# result for 1000 iterations trial 1 = 12.662 mean --> 13 is most common