#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: witkowski
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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

mean_fcasts = np.mean(fcasts, axis=0)
median_fcasts = np.median(fcasts, axis=0)

outcome = np.array([1,1,0,1,1,0,0,1])

def qsr(y,x):
    return 1-(y-x)**2

def extremize(vec,a):
    return vec**a/(vec**a+(1-vec)**a)

def qsr_forecaster_sets(i, forecasts, outcomes):
    return qsr(forecasts[i,:],outcomes).mean()


n = fcasts.shape[0]
    
# print("Ranking according to QSR:", np.argsort(-np.array([qsr_forecaster(i) \
#                                                          for i in range(n)]))+1)
# print("Ranking according to LSR:", np.argsort(-np.array([lsr_forecaster(i)\
#                                                          for i in range(n)]))+1)
# for a in range(10,101):
#       print("Extremized median with a =", a/10, ":",qsr(extremize(median_fcasts,a/10), outcome).mean())

# for a in range(10,101):
#       print("Extremized mean with a =", a/10, ":",qsr(extremize(mean_fcasts,a/10), outcome).mean())

n = fcasts.shape[0]

l = 7
k = 20
a = 2.0
   

avg_all_mean = 0
avg_topk_mean = 0
avg_all_mean_ext = 0
avg_topk_mean_ext = 0

avg_all_median = 0
avg_topk_median = 0
avg_all_median_ext = 0
avg_topk_median_ext = 0


iterations = 1000
for i in range(iterations):
    fcasts_train, fcasts_test, outcome_train, outcome_test = train_test_split(fcasts.T, outcome, train_size=l)
    fcasts_train = fcasts_train.T
    fcasts_test = fcasts_test.T
    
    ranked_forecasters_train = np.argsort(-np.array([qsr_forecaster_sets(i, fcasts_train, outcome_train) for i in range(n)]))
    topk_forecasters_train = ranked_forecasters_train[:k]
    
    mean_fcasts_topk = fcasts_test[topk_forecasters_train,:].mean(axis=0)
    median_fcasts_topk = np.median(fcasts_test[topk_forecasters_train,:], axis=0)
    
    avg_all_mean += qsr(fcasts_test.mean(axis=0), outcome_test).mean()
    avg_topk_mean += qsr(mean_fcasts_topk, outcome_test).mean()
    avg_all_mean_ext += qsr(extremize(fcasts_test.mean(axis=0), a), outcome_test).mean()
    avg_topk_mean_ext += qsr(extremize(mean_fcasts_topk, a), outcome_test).mean()

    avg_all_median += qsr(np.median(fcasts_test, axis=0), outcome_test).mean()
    avg_topk_median += qsr(median_fcasts_topk, outcome_test).mean()
    avg_all_median_ext += qsr(extremize(np.median(fcasts_test, axis=0), a), outcome_test).mean()
    avg_topk_median_ext += qsr(extremize(median_fcasts_topk, a), outcome_test).mean()

print("Brier score on test (all forecasters, mean)  :", np.round(avg_all_mean/iterations, 4))
print("Brier score on test (top k forecasters, mean):", np.round(avg_topk_mean/iterations, 4))
print("Brier score on test (all forecasters, extremized mean)  :", np.round(avg_all_mean_ext/iterations, 4))
print("Brier score on test (top k forecasters, extremized mean):", np.round(avg_topk_mean_ext/iterations, 4))

print("Brier score on test (all forecasters, median)  :", np.round(avg_all_median/iterations, 4))
print("Brier score on test (top k forecasters, median):", np.round(avg_topk_median/iterations, 4))
print("Brier score on test (all forecasters, extremized median)  :", np.round(avg_all_median_ext/iterations, 4))
print("Brier score on test (top k forecasters, extremized median):", np.round(avg_topk_median_ext/iterations, 4))




def selection_prob(fcast_vec, outcome):
    scores = qsr(fcast_vec, outcome)
    print(scores)
    n = len(fcast_vec)
    
    f = np.zeros(n)
    
    for i in range(n):
        f[i] = 1/n - 1/n * (scores[i] - (scores.sum()-scores[i])/(n-1))
    return f




a = np.array([1,2,3])/6

print(selection_prob(a, 1))





















