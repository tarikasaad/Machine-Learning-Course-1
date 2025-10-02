import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import BaggingRegressor

breast_cancer = datasets.load_breast_cancer()
features = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)
targets = breast_cancer.target
























