import pandas as pd
import numpy as np

fcasts_raw = pd.read_csv("fcasts_raw.csv", index_col=0)
rows, cols = fcasts_raw.shape

# Impute 50/50 for all forecaster-question combinations that do not sum to 1.

for i in range(rows):
    for j in range(0, cols, 2):
        if fcasts_raw.iloc[i, j] + fcasts_raw.iloc[i, j + 1] != 100:
            fcasts_raw.iloc[i, j], fcasts_raw.iloc[i, j + 1] = 50, 50

fcasts = np.array(fcasts_raw.iloc[:, [1, 3, 5, 7, 9, 11, 13, 15]]) / 100

outcomes = np.array([0, 1, 1, 0, 1, 0, 0, 1])


# Brier scoring:
def qsr(y, x):
    return 1 - (y - x) ** 2

def qsr_forecaster(i):
    return np.mean(qsr(fcasts[i, :], outcomes))


# With list comprehensions:
# print("Ranking according to QSR:", np.argsort(-np.array([qsr_forecaster(i) for i in range(rows)])))

# Without list comprehensions:
scores_brier = np.zeros(rows)

for i in range(rows):
    scores_brier[i] = qsr_forecaster(i)

print("Ranking according to QSR (from best to worst):", np.argsort(-scores_brier))


# Logarithmic scoring:
def lsr(y, x):
    if x == 1:
        return np.log(y)
    else:
        return np.log(1 - y)


def lsr_safe(y, x):
    if (y == 0 and x == 1) or (y == 1 and x == 0):
        return np.NINF
    else:
        return lsr(y, x)


def lsr_forecaster(i):
    number_questions = len(outcomes)
    tmp = 0
    for j in range(number_questions):
        tmp = tmp + lsr_safe(fcasts[i, j], outcomes[j])
    return tmp / number_questions


# With list comprehensions:
# print("Ranking according to LSR:", np.argsort(-np.array([lsr_forecaster(i) for i in range(rows)])))

# Without  list comprehensions:
scores_log = np.zeros(rows)

for i in range(rows):
    scores_log[i] = lsr_forecaster(i)

print("Ranking according to LSR (from best to worst):", np.argsort(-scores_log))


def extremize(vec, a):
    return vec ** a / (vec ** a + (1 - vec) ** a)


number_questions = len(outcomes)

print("Average Brier score over all forecasters:                      ", np.mean(scores_brier))
print("Brier score when reporting 50/50 on all questions:             ", 0.75)
print("Brier score of mean of all forecasters' forecasts:             ",
      np.mean(qsr(np.mean(fcasts, axis=0), outcomes)))
print("Brier score of median of all forecasters' forecasts:           ",
      np.mean(qsr(np.median(fcasts, axis=0), outcomes)))
print("Brier score of extremized mean of all forecasters' forecasts:  ",
      np.mean(qsr(extremize(np.mean(fcasts, axis=0), 2), outcomes)))
print("Brier score of extremized median of all forecasters' forecasts:",
      np.mean(qsr(extremize(np.median(fcasts, axis=0), 2), outcomes)))

def selection_prob(fcast_vec, single_outcome):
    n = len(fcast_vec)
    scores = qsr(fcast_vec, single_outcome)      #way to do it without a for loop
    #scores = np.zeros(n)
    #for i in range(n):
        #scores[i] = qsr(fcast_vec[1], single_outcome)
    f = np.zeros(n)

    return.....