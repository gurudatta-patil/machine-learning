#3
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

def sigmoid(x):
    sig=1 / (1 + np.exp(-x))
    return sig

def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    for i in range(iterations):
        theta -= learning_rate*(((sigmoid(theta.transpose()@X.transpose())-y.transpose())@X).transpose())
    return theta

df = pd.read_csv("emails.csv")
X = df.iloc[:, 1:-1]
y_temp = df.iloc[:, -1]   
kf = KFold(n_splits=5)
fold = 1
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y_temp.iloc[train_index], y_temp.iloc[test_index]
    y = np.zeros((4000, 1))
    for i in range(0,4000):
        y[i][0]=y_train.iloc[i]
    theta = np.zeros((X_train.shape[1],1))
    learning_rate = 0.03
    iterations = 1000
    theta= gradient_descent(X_train, y, theta, learning_rate, iterations)
    y_pred = sigmoid(X_test @ theta)
    y_pred[0]=[1 if x >= 0.5 else 0 for x in y_pred[0]]
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print("Fold", fold)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print()
    fold += 1

