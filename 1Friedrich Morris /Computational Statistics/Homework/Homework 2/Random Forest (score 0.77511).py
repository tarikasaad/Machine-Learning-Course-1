# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 16:15:08 2023

@author: 49163
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the training and test data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Calculate the survival rates for women and men
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women) / len(women)

men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men) / len(men)

print("% of women who survived:", rate_women)
print("% of men who survived:", rate_men)

# Prepare the data for training
y = train_data["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

# Create and train the Random Forest Classifier model
model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=1)  # You can adjust these hyperparameters
model.fit(X, y)
predictions = model.predict(X_test)

# Output the results to a CSV file
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('rf_submission.csv', index=False)
print("Your Random Forest submission was successfully saved!")
