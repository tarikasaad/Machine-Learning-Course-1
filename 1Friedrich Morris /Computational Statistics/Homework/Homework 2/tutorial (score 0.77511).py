# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 15:28:24 2023

@author: 49163
"""

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
predictions = model.predict(X_test)



# output results:

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('tutorial_submission.csv', index=False)
print("Your submission was successfully saved!")



# what does get_dummies do?

# 'pd.get_dummies()' takes takes the values from the column(s) selected and creates
# new columns for all unique values and fills the rows with 1's or 0's, to indicate
# which row has which value, kind of like a True and False statement, which allows
# the computer understand the values in text format as well, by converting text into
# binary values.

print("what does 'get_dummies' do?")
print()
print("'pd.get_dummies()' takes takes the values from the column(s) selected and creates new columns for all unique values and fills the rows with 1's or 0's, to indicate which row has which value, kind of like a True and False statement, which allows the computer understand the values in text format as well, by converting text into binary values.")

