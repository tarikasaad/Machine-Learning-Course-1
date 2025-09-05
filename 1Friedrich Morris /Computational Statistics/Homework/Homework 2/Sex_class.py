# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 19:32:15 2023

@author: 49163
"""

# Random Forest Classifier based on sex survial percentage based on class (score on kaggle = 0.76794)

# import data:

# import data:

import pandas as pd

train_data = pd.read_csv("train.csv")
train_data.head()

test_data = pd.read_csv("test.csv")
test_data.head()

# Feature Engineering: Create 'FamilySize' feature
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch']
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch']

# Data Cleaning: Fill blank age values with the mean age of the dataset
mean_age = train_data["Age"].mean()
train_data["Age"].fillna(mean_age, inplace=True)
test_data["Age"].fillna(mean_age, inplace=True)

# Define survival rates based on gender, class, and family size
survival_rates = {
    'female': {1: 0.9681, 2: 0.9211, 3: 0.5},
    'male': {1: 0.3689, 2: 0.1574, 3: 0.1354}
}

# Create a function to calculate survival based on the specified criteria
def calculate_survival(row):
    gender = row['Sex']
    pclass = row['Pclass']
    
    if gender in survival_rates and pclass in survival_rates[gender]:
        return survival_rates[gender][pclass]
    else:
        return 0  # Default to 0 for cases not specified

# Apply the calculate_survival function to the DataFrame to get survival predictions
train_data['Survival_Rate'] = train_data.apply(calculate_survival, axis=1)
test_data['Survival_Rate'] = test_data.apply(calculate_survival, axis=1)

# create model to predict if passenger survives:
from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]

# Use the calculated Survival_Rate and FamilySize as features
features = ["Age", "Survival_Rate", "FamilySize"]
X = train_data[features]
X_test = test_data[features]

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

# output results:
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('sex_class_submission.csv', index=False)
print("Your submission was successfully saved!")
