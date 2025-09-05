# -*- coding: utf-8 -*-



print()
print("author: Robert Smith")
print("Group: Robert, Joaquin, Louis")
print()
print()
print("Homework 1")
print("Part 2")
print()
print()
print()


# from linear_regression.py:

# Importing necessary libraries
import numpy as np # NumPy for numerical operations
import matplotlib.pyplot as plt # Matplotlib for plotting
from sklearn.linear_model import LinearRegression # Linear regression model from scikit-learn

# data points:
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 2.5]) # Independent variable
y = np.array([7.6, 5.0, 4.0, 3.5, 4.0, 5.0, 7.0, 4.0]) # Dependent variable

# Scatter plot of the data points
plt.scatter(x,y)

# Creating an instance of Linear Regression model
polyreg = LinearRegression() 
 
# Reshaping x to a column vector
X = x.reshape(-1, 1)

# Fitting the linear regression model
polyreg.fit(X,y)

# Predicting y for a new value of x (10)
new_X = np.array([[10]])
polyreg.predict(new_X)

# Predicting y for multiple new values of x (-10, 15, 50)
new_X = np.array([[-10],[15],[50]])
polyreg.predict(new_X)

# Generating a sequence of x values for prediction
X_seq = np.linspace(0,10,11).reshape(-1,1)
polyreg.predict(X_seq)



# a, b)

# Function to transform the feature matrix by adding quadratic features
def transformed_feature_matrix_quadratic(X):
    X_quadratic = np.append(X, X**2, axis = 1) # Appending the square of X to the original feature matrix
    return X_quadratic

# Function to transform the feature matrix by adding cubic features
def transformed_feature_matrix_singlecube(X):
    X_cubed = np.append(X, X**3, axis = 1) # Appending the cube of X to the original feature matrix
    X_cubed_array = np.array(X_cubed[:, 1]) # Extracting the second column (X^3) as a separate array
    return X_cubed_array.reshape(-1, 1 )  # Reshaping to a column vector
    

# Transforming the feature matrix by adding quadratic features 
X_quadratic = transformed_feature_matrix_quadratic(X) 
# Fitting the linear regression model with quadratic features
polyreg.fit(X_quadratic, y)

# Generating a sequence of x values with quadratic features for prediction
X_seq_quadratic = np.append(X_seq, X_seq**2, axis = 1)

# Plotting the predictions with quadratic features in green
plt.plot(X_seq, polyreg.predict(X_seq_quadratic), color="green") 
 
# Appending the cubic features to the original feature matrix 
X_cubed = np.append(X, X**3, axis = 1)

X_cubed_array = transformed_feature_matrix_singlecube(X)

# Fitting the linear regression model with cubic features
polyreg.fit(X_cubed, y)

# Appending cubic features to the sequence of x values for prediction
X_seq_cubed = np.append(X_seq, X_seq**3, axis = 1)

# Plotting the predictions with cubic features in red
plt.plot(X_seq, polyreg.predict(X_seq_cubed), color="red")

# plt.ylim([3,8.5]), plt.xlim(0,8.5)

print("a)")
print()
print()
print(transformed_feature_matrix_quadratic(X))
print()
print(" * To view result of X_quadratic as an array, please input 'X_quadratic' into the terminal")
print()
print()
print("b)")
print()
print()
print(transformed_feature_matrix_singlecube(X))
print()
print(" * To view result of X_cubed_array as an array, please input 'X_cubed_array' into the terminal")
print()
print()
print()



# c)

print("c)")
print()
print()

# Function to calculate Mean Squared Error (MSE) of a model
def mse(model, X, y):
    return ((y - model.predict(X))**2).mean()   # Calculating the mean squared difference between actual and predicted values

def transformed_feature_matrix_singlemin(X):
    # Experiment with different transformations here until you find one that achieves a training MSE < 0.06
    # For example, you can try various power transformations or other non-linear functions
    X_singlemin = np.append(X, X**2, axis = 1) # Apply the transformation to X
    return X_singlemin

# Print the transformed feature matrix:
print(transformed_feature_matrix_singlemin(X))
print()
print(" * To view result of transformed_feature_matrix_singlemin(X) as an array, please input 'X_singlemin' into the terminal")
print()

# Applying custom transformation to X:
X_singlemin = transformed_feature_matrix_singlemin(X)




# calculate and print mse for regression against transformed feature matrix:
polyreg_singlemin = LinearRegression()
polyreg_singlemin.fit(X_singlemin, y)
print("MSE of own model:", mse(polyreg_singlemin, X_singlemin, y))


# plotting regression on graph:
    
# Generating a sequence of x values for prediction:
X_seq_singlemin = np.linspace(0, 10, 11).reshape(-1, 1)

# Transforming the X_seq_singlemin using the same transformation:
X_seq_singlemin_transformed = transformed_feature_matrix_singlemin(X_seq_singlemin)

# Predicting y values for the transformed X_seq_singlemin:
y_seq_singlemin_pred = polyreg_singlemin.predict(X_seq_singlemin_transformed)

# Plotting the predictions with the custom transformation in blue:
plt.plot(X_seq_singlemin, y_seq_singlemin_pred, color="blue")

plt.show()



# Concluding Questions:
# 1) Provide a short discussion of your result: How did you find it?
# Our choice of transformation impacts the model's performance as it tries to capture a more
# complex relationships between the independent and dependent variables.

# 2) What is the intuition behind it?
# Our transformation is a manual approach to improve the model's fit, however since it is not
# always intuitive, did a lot of trial and error until we found it.

# 3) What might be problematic about this from an applied data science perspective?
# From an applied data science perspective, this transformation approach might be problematic
# for large datasets as it can be time-consuming and costly and might not fit well with new data.



# printing answers:
print()
print()
print()
print("Concluding Questions:")
print()
print()
print("1) Provide a short discussion of your result: How did you find it?")
print()
print("Our choice of transformation impacts the model's performance as it tries to capture a more complex relationships between the independent and dependent variables.")
print()
print()
print("2) What is the intuition behind it?")
print()
print("Our transformation is a manual approach to improve the model's fit, however since it is not always intuitive, did a lot of trial and error until we found it.")
print()
print()
print("3) What might be problematic about this from an applied data science perspective?")
print()
print("From an applied data science perspective, this transformation approach might be problematic for large datasets as it can be time-consuming and costly and might not fit well with new data.")
print()


