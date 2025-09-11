import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures  
from sklearn.pipeline import make_pipeline 
from sklearn.model_selection import train_test_split

wage = pd.read_csv('Theory /Supervised Learning/Wage.csv')

def mse(model,X,y):
    return np.mean((y - model.predict(X))**2)

# -------------------
# Exercise Numer 1
# -------------------

X = np.araay(wage.[:, "age"]).reshape(-1,1) #large X because it is a matrix 
y = np.array(wage.[:, "wage"]) #single dimensional therefore no reshape needed
X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.1, random_state=10)

plt.scatter(X_train,y_train, color="black")
plt.scatter(X_validate, y_validate, color="grey")

# -------------------
# Exercise Numer 2
# -------------------
#why do we dont the reshape
model1 = make_pipeline(PolynomialFeatures(degree = 1), LinearRegression())
model1.fit(X_train,y_train)

X_new_one = ([[0]])
model1.predict(X_new_one)

X_new_two = ([[100]])
model1.predict(X_new_two)

X_seq = np.linspace(15,85,101).reshape(-1,1)
plt.plot(X_seq, model1.predict(X_seq), color="blue")
plt.show()

print(mse(model1, X_validate, y_validate))

# -------------------
# Exercise Numer 3: 
# -------------------

model5 = make_pipeline(PolynomialFeatures(degree=5), LinearRegression())
model5.fit(X_train, y_train)
plt.plot(X_seq, model5.predict(X_seq), color="red")
print(mse(model5, X_validate, y_validate))

# -------------------
# Exercise Numer 3: 
# -------------------

model12 = make_pipeline(PolynomialFeatures(), LinearRegression())
model12.fit(X_train, y_train)
plt.plot(X_seq, model12.predict(X_seq, color="green"))
print(mse(model5, X_validate, y_validate))

# -------------------
# Exercise Numer 6: Line would still look very similar and high degree polynomial would go crazy, which would lead them to 
# bad training validation
# The more data you have the safer you can fit compexer models 
# -------------------

# -------------------
# Exercise Numer 6: (looked at validation MSE): 12 should be lowest. You can just ignore all the high degree coefficients
# -------------------