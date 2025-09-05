# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:09:01 2023

@author: 49163
"""


# random forest, but female survive, but male don't (score on kaggle = 0.76555)
# import data:
    
import pandas as pd

train_data = pd.read_csv("train.csv")
train_data.head()

test_data = pd.read_csv("test.csv")
test_data.head()

# create calculations of people survived:

women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of women who survived:", rate_women)
print("% of men who survived:", rate_men)

# create model to predict if passenger survives:

from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)

# Predict survival values and set them for males and females
predictions = model.predict(X_test)
test_data['Survived'] = 0  # Default to 0 for all passengers
test_data.loc[test_data['Sex'] == 'female', 'Survived'] = 1  # Set survival to 1 for females

# Output results:

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': test_data['Survived']})
output.to_csv('gender_submission.csv', index=False)
print("Your submission was successfully saved!")


