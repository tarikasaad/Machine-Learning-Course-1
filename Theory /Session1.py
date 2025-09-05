import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression #Only importing a sub-part of the module


x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 2.5])
y = np.array([7.6, 5.0, 4.0, 3.5, 4.0, 5.0, 7.0, 4.0])

polyreg = LinearRegression()

X = x.reshape(-1,1) #-1 says do what ever you need to do to make it work on the other axis where we have the 1 #We want it reshaped because skikit learn expectst a 2D Array
polyreg.fit(X,y)

new_X = np.array(([10])) #still two dimensional (np.ndim(new_x) to check dimension)
print(polyreg.predict(new_X))

new_X = np.array(([0])) 
print(polyreg.predict(new_X2))

#only need those two data points for a line
#for a quadratic you need three data points

#You can check for multiple values
#new_X = np.array([[-10],[15],[50]])
#polyreg.predict(new_X)

#You can also check for a range of numbers
#X_seq = np.linspace(0,10,101).reshape(-1,1) #Why the 101? Has something to do with 0 to 10 inculded/not included
#polyreg.predict(X_seq)

plt.scatter(x,y)

plt.plot(X_seq, polyreg.predict(X_seq), color="black") #plt.plot(x-values, y-values, type)

plt.show()

def mse(model,X,y):
    return np.mean((y - model.predict(X))**2)
