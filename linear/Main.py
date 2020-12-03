import math
import numpy as np
import matplotlib.pyplot as plt
import os

def read_data(file, m):
    n = int(file.readline().strip())
    X = np.zeros((n, m + 1))
    y = np.zeros(n)
    for i in range(n):
        line = list(map(float, file.readline().strip().split()))
        for j in range(m):
            X[i][j] = line[j]
        X[i][m] = 1
        y[i] = line[m]
    return X, y


def read_dataset(file):
    m = int(file.readline().strip())
    X_train, y_train = read_data(file, m)
    X_test, y_test = read_data(file, m)
    return X_train, y_train, X_test, y_test


def SMAPE(model, X_test, Y_test):
    n, *_ = X_test.shape
    return sum([abs((x @ model) - y) / (abs(x @ model) + abs(y)) for (x, y) in zip(X_test, Y_test)]) / n


def get_inverse_model(X_train, y_train, tau=0):
    n, m = X_train.shape
    F = X_train
    F_t = np.transpose(F)
    to_inv = (F_t @ F) + tau * np.eye(m)
    model_t = np.linalg.inv(to_inv) @ F_t @ y_train
    return model_t.T


def matrix_solution(X_train, y_train, X_test, y_test):
    model = get_inverse_model(X_train, y_train, 5)
    return SMAPE(model, X_test, y_test)


def run_test(filename):
    print('Processing file {0}'.format(filename))
    with open('{0}'.format(filename), 'r') as file:
        X_train, y_train, X_test, y_test = read_dataset(file)
        err = matrix_solution(X_train, y_train, X_test, y_test)
        return err


suites = ["tests/"]

for suite in suites:
    suite_sum = 0
    cnt = 0
    for filename in sorted(os.listdir(suite)):
        err = run_test(suite + filename)
        cnt += 1
        print("Result = {}".format(err))
        suite_sum += err
    score = cnt - suite_sum
    print("Overall score for suite \"{0}\" is {1} / {2}".format(suite, score, cnt))
