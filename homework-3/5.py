#5
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
from threading import Thread
from statistics import fmean
import warnings

warnings.filterwarnings('ignore')
def euclidean_dist(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn_prob(X_train, Y_train, X_test, k=5):
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
        probability = fmean(total)
        y_pred.append(probability)
    return np.array(y_pred)

df = pd.read_csv("emails.csv")
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Y_prob = knn_prob(X_train, y_train, X_test)
fpr_knn, tpr_knn, _ = roc_curve(y_test, Y_prob)
auc_knn = auc(fpr_knn, tpr_knn)

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
X_train, X_test, y_train, y_test = train_test_split(X, y_temp, test_size=0.2, random_state=42)
y = np.zeros((4000, 1))
for i in range(0,4000):
    y[i][0]=y_train.iloc[i]
theta = np.zeros((X_train.shape[1],1))
learning_rate = 0.05
iterations = 1000
theta= gradient_descent(X_train, y, theta, learning_rate, iterations)
y_prob = X_test @ theta
fpr_log, tpr_log, _ = roc_curve(y_test, y_prob)
auc_log = auc(fpr_log, tpr_log)
plt.figure()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.plot(fpr_knn, tpr_knn, label='kNN (area = %0.2f)' % auc_knn)
plt.plot(fpr_log, tpr_log, label='Logistic Regression (area = %0.2f)' % auc_log)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.plot([0, 1], [0, 1], 'k--')
plt.legend(loc="lower right")
plt.show()
