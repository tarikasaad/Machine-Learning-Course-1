import numpy as np
import pandas as pd

np.random.seed(10)

mean_A = [1,1]
mean_B = [2,2]
cov = np.eye(2)
n =10

df = np.random.multivariate_normal(mean_A, cov, n)

df2 = 