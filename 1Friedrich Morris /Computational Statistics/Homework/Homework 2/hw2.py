# -*- coding: utf-8 -*-
'''
Group name: "Will think about it later"
Group memebers: Joaquin Rodriguez, Robert Smith, Louis Schmidt
'''
print()
print('Group name: "Will think about it later"')
print("Group memebers: Joaquin Rodriguez, Robert Smith, Louis Schmidt")
print()
print()
print("Homework 2")
print()
print()
print()
print("what does 'get_dummies' do from the Titanic Tutorial?")
print()
print("'pd.get_dummies()' takes takes the values from the column(s) selected and creates new columns for all unique values and fills the rows with 1's or 0's, to indicate which row has which value, kind of like a True and False statement, which allows the computer understand the values in text format as well, by converting text into binary values.")
print()
print()
print()
print()



# best model at the bottom



# =============================================================================
# polynomial regressions of degree 2
# polynomial regression of degree 3
# regression tree of depth 5
# 
# # import pandas as pd
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import train_test_split
# import numpy as np
# 
# # Load the training and test data
# train_data = pd.read_csv("train.csv")
# test_data = pd.read_csv("test.csv")
# 
# # Prepare the data for training
# y = train_data["Survived"]
# features = ["Pclass", "Sex", "SibSp", "Parch"]
# X = pd.get_dummies(train_data[features])
# X_test = pd.get_dummies(test_data[features])
# 
# # Split the training data into train and validation sets
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
# 
# # Create a dictionary to store MSE results and predictions
# results = {}
# 
# # Polynomial Regression (Degree 2)
# poly = PolynomialFeatures(degree=2)
# X_poly_train = poly.fit_transform(X_train)
# X_poly_val = poly.transform(X_val)
# poly_reg = LinearRegression()
# poly_reg.fit(X_poly_train, y_train)
# poly_predictions = poly_reg.predict(X_poly_val)
# poly_mse = mean_squared_error(y_val, poly_predictions)
# results["Polynomial Regression (Degree 2)"] = {
#     "mse": poly_mse,
#     "predictions": np.round(poly_reg.predict(poly.transform(X_test)))
# }
# 
# # Polynomial Regression (Degree 3)
# poly = PolynomialFeatures(degree=3)
# X_poly_train = poly.fit_transform(X_train)
# X_poly_val = poly.transform(X_val)
# poly_reg = LinearRegression()
# poly_reg.fit(X_poly_train, y_train)
# poly_predictions = poly_reg.predict(X_poly_val)
# poly_mse = mean_squared_error(y_val, poly_predictions)
# results["Polynomial Regression (Degree 3)"] = {
#     "mse": poly_mse,
#     "predictions": np.round(poly_reg.predict(poly.transform(X_test)))
# }
# 
# # Regression Tree (Depth 5)
# tree_reg = DecisionTreeRegressor(max_depth=5)
# tree_reg.fit(X_train, y_train)
# tree_predictions = tree_reg.predict(X_val)
# tree_mse = mean_squared_error(y_val, tree_predictions)
# results["Regression Tree (Depth 5)"] = {
#     "mse": tree_mse,
#     "predictions": np.round(tree_reg.predict(X_test))
# }
# 
# # Print the MSE results for all configurations
# print("\nMSE Results:")
# for model_name, data in results.items():
#     print(f"{model_name} - MSE: {data['mse']}")
# 
# # Save predictions to CSV for each model
# for model_name, data in results.items():
#     output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': data['predictions']})
#     output.to_csv(f'{model_name}_submission.csv', index=False)
#     print(f"Predictions for {model_name} were successfully saved to {model_name}_submission.csv")
# =============================================================================



# =============================================================================
# # Decision Tree Classifier (score on kaggle = 0.72248)
# 
# # import data:
# 
# import pandas as pd
# 
# train_data = pd.read_csv("train.csv")
# train_data.head()
# 
# test_data = pd.read_csv("test.csv")
# test_data.head()
# 
# # Data Cleaning: Fill missing age values with the mean age of all passengers
# mean_age = train_data["Age"].mean()
# train_data["Age"].fillna(mean_age, inplace=True)
# test_data["Age"].fillna(mean_age, inplace=True)
# 
# # create model to predict if passenger survives:
# 
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.impute import SimpleImputer
# 
# y = train_data["Survived"]
# 
# features = ["Pclass", "Sex", "Age", "SibSp", "Parch"]
# X = pd.get_dummies(train_data[features])
# 
# # Use SimpleImputer to handle missing values in the test set as well
# X_test = pd.get_dummies(test_data[features])
# imputer = SimpleImputer(strategy="mean")
# X_test = pd.DataFrame(imputer.fit_transform(X_test), columns=X_test.columns)
# 
# model = DecisionTreeClassifier(random_state=1)
# model.fit(X, y)
# 
# # Predict survival values:
# predictions = model.predict(X_test)
# 
# # Output results:
# output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
# output.to_csv('Tree_submission.csv', index=False)
# print("Your submission was successfully saved!")
# =============================================================================




# =============================================================================
# # random forest with survial rules (score on Kaggle = 0.63157)
# 
# # thought process:
# # Females are more likely to survive and first class almost always survives,
# # regardless of age, second class almost half, but less the older they get 
# # and almost nobody of third class and I'd say in general younger than 16
# # survives more often than over.
# 
# # Data Cleaning:
# # As age does not always have a value, cleaning by giving these the
# # mean age of all passengers.
# 
# 
# # import data:
# 
# import pandas as pd
# 
# train_data = pd.read_csv("train.csv")
# train_data.head()
# 
# test_data = pd.read_csv("test.csv")
# test_data.head()
# 
# # Data Cleaning: Fill missing age values with the mean age of all passengers
# mean_age = train_data["Age"].mean()
# train_data["Age"].fillna(mean_age, inplace=True)
# test_data["Age"].fillna(mean_age, inplace=True)
# 
# # create model to predict if passenger survives:
# 
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.impute import SimpleImputer
# 
# y = train_data["Survived"]
# 
# features = ["Pclass", "Sex", "Age", "SibSp", "Parch"]
# X = pd.get_dummies(train_data[features])
# 
# # Use SimpleImputer to handle missing values in the test set as well
# X_test = pd.get_dummies(test_data[features])
# imputer = SimpleImputer(strategy="mean")
# X_test = pd.DataFrame(imputer.fit_transform(X_test), columns=X_test.columns)
# 
# model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
# model.fit(X, y)
# 
# # Predict survival values based on your rules:
# test_data['Survived'] = 0  # Default to 0 for all passengers
# 
# # Rule 1: Females are more likely to survive
# test_data.loc[test_data['Sex'] == 'female', 'Survived'] = 1
# 
# # Rule 2: First-class passengers almost always survive
# test_data.loc[test_data['Pclass'] == 1, 'Survived'] = 1
# 
# # Rule 3: Second-class passengers under a certain age survive
# test_data.loc[(test_data['Pclass'] == 2) & (test_data['Age'] < 30), 'Survived'] = 1
# 
# # Rule 4: Younger individuals (under 18) survive more often
# test_data.loc[test_data['Age'] < 18, 'Survived'] = 1
# 
# # Output results:
# output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': test_data['Survived']})
# output.to_csv('rule_submission.csv', index=False)
# print("Your submission was successfully saved!")
# =============================================================================



# =============================================================================
# # Random Forest Classifier based on sex survial percentage based on class (score on kaggle = 0.76794)
# 
# # import data:
# 
# 
# import pandas as pd
# 
# train_data = pd.read_csv("train.csv")
# train_data.head()
# 
# test_data = pd.read_csv("test.csv")
# test_data.head()
# 
# # Feature Engineering: Create 'FamilySize' feature
# train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch']
# test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch']
# 
# # Data Cleaning: Fill blank age values with the mean age of the dataset
# mean_age = train_data["Age"].mean()
# train_data["Age"].fillna(mean_age, inplace=True)
# test_data["Age"].fillna(mean_age, inplace=True)
# 
# # Define survival rates based on gender, class, and family size
# survival_rates = {
#     'female': {1: 0.9681, 2: 0.9211, 3: 0.5},
#     'male': {1: 0.3689, 2: 0.1574, 3: 0.1354}
# }
# 
# # Create a function to calculate survival based on the specified criteria
# def calculate_survival(row):
#     gender = row['Sex']
#     pclass = row['Pclass']
#     
#     if gender in survival_rates and pclass in survival_rates[gender]:
#         return survival_rates[gender][pclass]
#     else:
#         return 0  # Default to 0 for cases not specified
# 
# # Apply the calculate_survival function to the DataFrame to get survival predictions
# train_data['Survival_Rate'] = train_data.apply(calculate_survival, axis=1)
# test_data['Survival_Rate'] = test_data.apply(calculate_survival, axis=1)
# 
# # create model to predict if passenger survives:
# from sklearn.ensemble import RandomForestClassifier
# 
# y = train_data["Survived"]
# 
# # Use the calculated Survival_Rate and FamilySize as features
# features = ["Age", "Survival_Rate", "FamilySize"]
# X = train_data[features]
# X_test = test_data[features]
# 
# model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
# model.fit(X, y)
# predictions = model.predict(X_test)
# 
# # output results:
# output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
# output.to_csv('sex_class_submission.csv', index=False)
# print("Your submission was successfully saved!")
# =============================================================================



# =============================================================================
# # random forest, but female survive, but male don't (score on kaggle = 0.76555)
# # import data:
#     
# import pandas as pd
# 
# train_data = pd.read_csv("train.csv")
# train_data.head()
# 
# test_data = pd.read_csv("test.csv")
# test_data.head()
# 
# # create calculations of people survived:
# 
# women = train_data.loc[train_data.Sex == 'female']["Survived"]
# rate_women = sum(women)/len(women)
# 
# men = train_data.loc[train_data.Sex == 'male']["Survived"]
# rate_men = sum(men)/len(men)
# 
# print("% of women who survived:", rate_women)
# print("% of men who survived:", rate_men)
# 
# # create model to predict if passenger survives:
# 
# from sklearn.ensemble import RandomForestClassifier
# 
# y = train_data["Survived"]
# 
# features = ["Pclass", "Sex", "SibSp", "Parch"]
# X = pd.get_dummies(train_data[features])
# X_test = pd.get_dummies(test_data[features])
# 
# model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
# model.fit(X, y)
# 
# # Predict survival values and set them for males and females
# predictions = model.predict(X_test)
# test_data['Survived'] = 0  # Default to 0 for all passengers
# test_data.loc[test_data['Sex'] == 'female', 'Survived'] = 1  # Set survival to 1 for females
# 
# # Output results:
# 
# output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': test_data['Survived']})
# output.to_csv('gender_submission.csv', index=False)
# print("Your submission was successfully saved!")
# =============================================================================



# =============================================================================
# Random Forest model with n_estimators=200 [score in Kaggle = 0.77511]
# 
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# 
# # Load the training and test data
# train_data = pd.read_csv("train.csv")
# test_data = pd.read_csv("test.csv")
# 
# # Calculate the survival rates for women and men
# women = train_data.loc[train_data.Sex == 'female']["Survived"]
# rate_women = sum(women) / len(women)
# 
# men = train_data.loc[train_data.Sex == 'male']["Survived"]
# rate_men = sum(men) / len(men)
# 
# print("% of women who survived:", rate_women)
# print("% of men who survived:", rate_men)
# 
# # Prepare the data for training
# y = train_data["Survived"]
# features = ["Pclass", "Sex", "SibSp", "Parch"]
# X = pd.get_dummies(train_data[features])
# X_test = pd.get_dummies(test_data[features])
# 
# # Create and train the Random Forest Classifier model
# model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=1)  # You can adjust these hyperparameters
# model.fit(X, y)
# predictions = model.predict(X_test)
# 
# # Output the results to a CSV file
# output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
# output.to_csv('rf_submission.csv', index=False)
# print("Your Random Forest submission was successfully saved!")
# =============================================================================



# =============================================================================
# Support Vector model [score in Kaggle = 0.76555] (score doesn't change, when C parameter was changed)
# 
# import pandas as pd
# from sklearn.svm import SVC
# 
# # load data
# train_data = pd.read_csv("train.csv")
# test_data = pd.read_csv("test.csv")
# 
# # Calculate the survival rates 
# women = train_data.loc[train_data.Sex == 'female']["Survived"]
# rate_women = sum(women)/len(women)
# 
# men = train_data.loc[train_data.Sex == 'male']["Survived"]
# rate_men = sum(men)/len(men)
# 
# print("% of women who survived:", rate_women)
# print("% of men who survived:", rate_men)
# 
# # prepare data
# y = train_data["Survived"]
# features = ["Pclass", "Sex", "SibSp", "Parch"]
# X = pd.get_dummies(train_data[features])
# X_test = pd.get_dummies(test_data[features])
# 
# # train svc model
# svc = SVC(kernel="linear", C=3)
# svc.fit(X, y)
# predictions = svc.predict(X_test)
# 
# # output the results
# output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
# output.to_csv('svc_submission.csv', index=False)  # Save the results to a CSV file
# print("Your SVC submission was successfully saved!")
# =============================================================================



# Best Model for submission Score in Kaggle [0.7799]

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


# Load data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Feature Engineering: Create 'FamilySize' feature
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch']
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch']

# Data Preparation
y = train_data["Survived"]  # Target variable
features = ["Pclass", "Sex", "SibSp", "Parch", "FamilySize"]  # Features used for training
X = pd.get_dummies(train_data[features])  # One-hot encode categorical features
X_test = pd.get_dummies(test_data[features])  # Preprocess test data similarly

# Initialize and train Random Forest model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)

# Cross-validation to evaluate model
cv_scores = cross_val_score(model, X, y, cv=5)

# Make predictions
predictions = model.predict(X_test)

# Create submission file
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('bestmodel.csv', index=False)
print("Your submission was successfully saved as 'bestmodel.csv!'")




# Explanation for Model Choice and Transformations
print()
print()
print()
print()
print("Why and what did we do?")
print()
print("We chose the Random Forest Classifier because it can be used for many datasets and is usually a very good model for classification tasks.")
print("We did some feature engineering/transformation, including creating a new feature we called 'FamilySize', to potentially get some additional information.")
print("We used One-hot encoding ('pd.get_dummies') to handle the categorical variables.")
print("The model and transformation are a result of experimentation, with various types of predictions and transformations to optimize the score.")
print()
