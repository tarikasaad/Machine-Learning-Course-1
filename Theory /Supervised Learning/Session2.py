import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures  #!new
from sklearn.pipeline import make_pipeline            #!new
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 2.5])
y = np.array([7.6, 5.0, 4.0, 3.5, 4.0, 5.0, 7.0, 4.0])

plt.scatter(x,y)
plt.show() 

polyreg = LinearRegression() 

X = x.reshape(-1,1) 
X_transformed = np.append(X, X**2, axis = 1)

polyreg.fit(X_transformed, y) 

X_seq = np.linspace(0,10,101).reshape(-1,1) 
X_seq_transformed = np.append(X_seq, X_seq**2, axis=1)

polyreg.predict(X_seq_transformed)
#plt.plot(X_seq, polyreg.predict(X_seq), color="black") 
plt.plot(X_seq, polyreg.predict(X_seq_transformed), color="black")
plt.show()

# ----------------
# Cubic Regression (with the use of PolynomialFeatures and make_pipeline)
# ----------------
#polyreg = LinearRegression()

X = x.reshape(-1,1) 
#X_transformed = np.append(X, X**2, axis = 1)
#X_transformed = np.append(X_transformed, X**3, axis=1)
polyreg=make_pipeline(PolynomialFeatures(degree = 5), LinearRegression())
polyreg.fit(X, y) 

X_seq = np.linspace(0,10,101).reshape(-1,1) 
plt.ylim(3,10) #to zoom in 
#X_seq_transformed = np.append(X_seq, X_seq**2, axis=1)
#X_seq_transformed = np.append(X_seq_transformed, X_seq**3, axis=1)

plt.plot(X_seq, polyreg.predict(X_seq), color="black")

#polyreg.predict(X_seq_transformed)
#plt.plot(X_seq, polyreg.predict(X_seq), color="black") 
#plt.plot(X_seq, polyreg.predict(X_seq_transformed), y, color="black")
#plt.show()

