# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 18:50:56 2023

@author: 49163
"""

import pandas as pd
import numpy as np

fcasts_raw = pd.read_csv("fcasts_raw.csv", index_col = 0)


rows, cols = fcasts_raw.shape

for i in range(0, cols, 2):
    for j in range(rows):
        if fcasts_raw.iloc[j,i] + fcasts_raw.iloc[j,i+1] != 100:
            tmp1 = fcasts_raw.iloc[j,i]
            tmp2 = fcasts_raw.iloc[j,i+1]
            
            fcasts_raw.iloc[j,i] = tmp1/(tmp1+tmp2)*100
            fcasts_raw.iloc[j,i+1] = tmp2/(tmp1+tmp2)*100
            
            
fcasts = np.array(fcasts_raw.iloc[:, [1,3,4,6,9,11,12,15]]/100)

outcome = np.array([1,1,0,1,1,0,0,1])

def qrs(y, x):
    return 1-(y-x)**2

def qsr_forecaster(i):
    return qrs(fcasts[i,:], outcome).mean()

ranking = []

for i in range(rows):
    ranking.append(qsr_forecaster(i))
    
print(np.argsort(-np.array(ranking))+1)

