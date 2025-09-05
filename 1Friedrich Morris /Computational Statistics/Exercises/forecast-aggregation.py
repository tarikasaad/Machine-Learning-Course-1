# -*- coding: utf-8 -*-
"""
@author: 49163
"""


import pandas as pd
import numpy as np

fcasts_raw = pd.read_csv("fcasts_raw.csv", index_col=0)
rows, cols = fcasts_raw.shape

for i in range(0,cols, 2):
    for j in range(rows):
        if fcasts_raw.iloc[j,i] + fcasts_raw.iloc[j,i+1] != 100:
            # print("Question:", i, "Forecaster:", j)
            tmp1 = fcasts_raw.iloc[j,i]
            tmp2 = fcasts_raw.iloc[j,i+1]
            fcasts_raw.iloc[j,i] = tmp1/(tmp1+tmp2)*100
            fcasts_raw.iloc[j,i+1] = tmp2/(tmp1+tmp2)*100
            

fcasts = np.array(fcasts_raw.iloc[:,[1,3,4,6,9,11,12,15]]/100)  # only "yes" prob mass of binary-outcome events

outcome = np.array([1,1,0,1,1,0,0,1])

def qsr(y,x):
    return 1-(y-x)**2

def qsr_forecaster(i):
    return qsr(fcasts[i,:],outcome).mean()


def extremize(vec,a):
    return (vec**a)/(vec**a+(1-vec)**a)


n = fcasts.shape[0]
    
# print("Ranking according to QSR:", np.argsort(-np.array([qsr_forecaster(i) \
#                                                         for i in range(n)]))+1)
    
print("Sorted scores of forecasters:\n", np.sort(np.array([qsr_forecaster(i) for i in range(n)])))
mean_fcasts = fcasts.mean(axis=0)
median_fcasts = np.median(fcasts, axis=0)
print("QSR of mean aggregation:", qsr(mean_fcasts, outcome).mean())
print("QSR of median aggregation:", qsr(median_fcasts, outcome).mean())
print("QSR of extremised mean aggregation:", qsr(extremize(mean_fcasts, 2), outcome).mean())
print("QSR of extremised median aggregation:", qsr(extremize(median_fcasts, 2), outcome).mean())

for a_int in range(10,101):
    print("Extremized mean with a=", a_int/10, "is:", qsr(extremize(mean_fcasts, a_int/10), outcome).mean())
    # print("Extremized median with a=", a_int/10, "is:", qsr(extremize(median_fcasts, 2), outcome).mean())

    






























