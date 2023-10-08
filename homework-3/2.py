#2
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from threading import Thread

def euclidean_dist(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn_pred(X_train, Y_train, X_test, k=1):
    y_pred = []
    for i in range(0,1000):
        x=X_test.iloc[i,:]
        all_dist = [] 
        for j in range(0,4000):
            x_train=X_train.iloc[j,:]
            y_train=Y_train.iloc[j]
            dist = euclidean_dist(x, x_train)
            all_dist.append((dist, y_train))
        all_dist.sort(key=lambda x: x[0])
        neighbors = all_dist[:k]
        total = [n[1] for n in neighbors]
        pred = max(set(total), key=total.count)
        y_pred.append(pred)
    return np.array(y_pred) 

df = pd.read_csv("emails.csv")
X = df.iloc[:, 1:-1] 
y = df.iloc[:, -1]
kf = KFold(n_splits=5)
fold = 1
for train_index, test_index in kf.split(X):
    print("Fold", fold)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = y.iloc[train_index], y.iloc[test_index]
    Y_pred = knn_pred(X_train, Y_train, X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print()
    fold += 1

