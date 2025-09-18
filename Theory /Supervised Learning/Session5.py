
#We need the stopping criteria for decision trees because otherwise it is overfitting 
#For every split you either stay the same or decrease your loss function 
# predictive value choose for regressions/ trees you finding the cut which minimizes the error 
#----------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

def plot_svc(svc, X, y, h=0.02, pad=0.25):
    x_min, x_max = X[:, 0].min()-pad, X[:, 0].max()+pad
    y_min, y_max = X[:, 1].min()-pad, X[:, 1].max()+pad
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.2)

    plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=plt.cm.Paired)
# Support vectors indicated in plot by vertical lines
    sv = svc.support_vectors_
    plt.scatter(sv[:,0], sv[:,1], c='k', marker='|', s=100, linewidths=1)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    print('Number of support vectors: ', svc.support_.size)

np.random.seed(10)
X_cluster1 = np.random.multivariate_normal((1,1), np.eye(2)*0.5, 10)
X_cluster2 = np.random.multivariate_normal((2,2), np.eye(2)*0.5, 10)

X = np.append(X_cluster1, X_cluster2, axis=0)

y = np.repeat([1,-1], 10)

# Plot the data using the plt.scatter function and colormap Paired.
# Is the data linearly separable?

plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Paired)

tree_classifier = DecisionTreeClassifier() #default value is gini method 
tree_classifier.fit(X,y)
#-------------------------------------
tree.plot_tree(tree_classifier) #Exercise 1
tree_classifier = DecisionTreeClassifier(max_leaf_nodes=6) #Exercise 2
tree_classifier = DecisionTreeClassifier(criterion= "entropy",max_leaf_nodes=3) #Exercise 3

x1_min, x2_min = np.min(X, axis=0)
x1_max, x2_max = np.max(X, axis=0)

grid_size = 100
x1_seq = np.linspace(x1_min, x1_max, grid_size) #remember: linspace is inclusive on bith sides 
x2_seq = np.linspace(x2_min, x2_max, grid_size)

X_seq = np.zeros((grid_size**2,2))

for x1 in range(grid_size):
    for x2 in range(grid_size):
        X_seq[x1*grid_size+x2] = x1_seq[x1], x2_seq[x2] #Exercise 4
        
predicted_labels = tree_classifier.predict(X_seq)
plt.scatter(X_seq[:,0], X_seq[:,1], s=0.1, c=predicted_labels, cmap=plt.cm.Paired)