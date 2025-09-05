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

def qsr_forecaster(i):
    return qsr(fcasts[i,:],outcome).mean()


n = fcasts.shape[0]

def selection_prob(fcast_vec, outcome):
    n = len(fcast_vec)
    scores = qsr(fcast_vec, outcome)
    f = np.zeros(n)
    for i in range(n):
        f[i] = 1/n + 1/n * (scores[i] - (scores.sum()-scores[i])/(n-1))
    return f

print("Selection probabilities for first question as vector:\n", selection_prob(fcasts[:,0], outcome[0]))

def overall_winner(fcasts_matrix, outcomes):
    n, m = fcasts_matrix.shape
    winners = np.zeros(n)
    for i in range(m):
        event_winner = np.random.choice(np.arange(n),p=selection_prob(fcasts_matrix[:,i], outcomes[i]))
        winners[event_winner] += 1
    candidates = np.flatnonzero(winners == np.max(winners))
    k = len(candidates)
    return np.random.choice(candidates, p=np.ones(k)/k)

def overall_winner_ranking(fcasts_matrix, outcomes):
    n, m = fcasts_matrix.shape
    winners = np.zeros(n)
    for j in range(10000):
        sample_overall_winner = overall_winner(fcasts_matrix, outcomes)
        winners[sample_overall_winner] += 1
    return np.argsort(-winners)


print("\nTruthful competition ranking:", overall_winner_ranking(fcasts,outcome))

print("Ranking according to QSR    :", np.argsort(-np.array([qsr_forecaster(i) \
                                                          for i in range(n)])))