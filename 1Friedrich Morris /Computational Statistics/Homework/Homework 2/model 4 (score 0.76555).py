# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 16:04:12 2023

@author: 49163
"""

import pandas as pd
from sklearn.svm import SVC

# load data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Calculate the survival rates 
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of women who survived:", rate_women)
print("% of men who survived:", rate_men)

# prepare data
y = train_data["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

# train svc model
svc = SVC(kernel="linear", C=3)  # You can adjust the C parameter for regularization
svc.fit(X, y)
predictions = svc.predict(X_test)

# output the results
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('svc_submission.csv', index=False)  # Save the results to a CSV file
print("Your SVC submission was successfully saved!")








