# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:31:38 2023

@author: 49163
"""


# random forest with survial rules (score on Kaggle = 0.63157)

# thought process:
# Females are more likely to survive and first class almost always survives,
# regardless of age, second class almost half, but less the older they get 
# and almost nobody of third class and I'd say in general younger than 16
# survives more often than over.

# Data Cleaning:
# As age does not always have a value, cleaning by giving these the
# mean age of all passengers.


# import data:

import pandas as pd

train_data = pd.read_csv("train.csv")
train_data.head()

test_data = pd.read_csv("test.csv")
test_data.head()

# Data Cleaning: Fill missing age values with the mean age of all passengers
mean_age = train_data["Age"].mean()
train_data["Age"].fillna(mean_age, inplace=True)
test_data["Age"].fillna(mean_age, inplace=True)

# create model to predict if passenger survives:

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

y = train_data["Survived"]

features = ["Pclass", "Sex", "Age", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])

# Use SimpleImputer to handle missing values in the test set as well
X_test = pd.get_dummies(test_data[features])
imputer = SimpleImputer(strategy="mean")
X_test = pd.DataFrame(imputer.fit_transform(X_test), columns=X_test.columns)

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)

# Predict survival values based on your rules:
test_data['Survived'] = 0  # Default to 0 for all passengers

# Rule 1: Females are more likely to survive
test_data.loc[test_data['Sex'] == 'female', 'Survived'] = 1

# Rule 2: First-class passengers almost always survive
test_data.loc[test_data['Pclass'] == 1, 'Survived'] = 1

# Rule 3: Second-class passengers under a certain age survive
test_data.loc[(test_data['Pclass'] == 2) & (test_data['Age'] < 30), 'Survived'] = 1

# Rule 4: Younger individuals (under 18) survive more often
test_data.loc[test_data['Age'] < 18, 'Survived'] = 1

# Output results:
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': test_data['Survived']})
output.to_csv('rule_submission.csv', index=False)
print("Your submission was successfully saved!")
