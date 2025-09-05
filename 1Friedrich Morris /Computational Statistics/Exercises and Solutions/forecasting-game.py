#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: witkowski
"""

import pandas as pd
import numpy as np

fcasts_raw = pd.read_csv("fcasts_raw.csv", index_col=0)
rows, cols = fcasts_raw.shape

for i in range(0,cols, 2):
    for j in range(rows):
        if fcasts_raw.iloc[j,i] + fcasts_raw.iloc[j,i+1] != 100:
            print("Question:", i, "Forecaster:", j)
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

def lsr(y,x):
    if x==1:
        return np.log(y)
    else:
        return np.log(1-y)

def lsr_safe(y,x):
    if (x==1 and y==0) or (x==0 and y==1):
        return np.NINF # Numpy constant: negative infinity
    else:
        return lsr(y,x)
        
def lsr_forecaster(i):
    number_questions = len(outcome)
    tmp = 0
    for question in range(number_questions):
        tmp = tmp + lsr_safe(fcasts[i,question],outcome[question])
    return tmp/number_questions

def lsr_forecaster2(i):
    return lsr_safe(fcasts[i,:],outcome).mean()
    

n = fcasts.shape[0]
    
print("Ranking according to QSR:", np.argsort(-np.array([qsr_forecaster(i) \
                                                         for i in range(n)]))+1)
print("Ranking according to LSR:", np.argsort(-np.array([lsr_forecaster(i)\
                                                         for i in range(n)]))+1)