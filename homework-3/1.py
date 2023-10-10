#1
import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt("D2z.txt")
X = data[:, :2]
y = data[:, 2] 
def euclidean_dist(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def predict(x, X, y):
    nn_label = nearest_neighbor(x, X, y)
    return nn_label

def nearest_neighbor(x, X, y):
    m_dist = np.inf
    m_index = -1
    for i in range(len(X)):
        dist = euclidean_dist(x, X[i])
        if dist < m_dist:
            m_dist = dist
            m_index = i
    return y[m_index]

x_min, x_max = -2, 2 
y_min, y_max = -2, 2 
h = 0.1
x_range=np.arange(x_min, x_max + h, h)
y_range=np.arange(y_min, y_max + h, h)
x_test=np.array([x for x in x_range])
y_test=np.array([y for y in y_range])
test_points=np.array([(x,y) for x in x_range for y in y_range])
Z = np.array([predict(x, X, y) for x in test_points])
plt.scatter(test_points[:, 0], test_points[:, 1], c=Z, marker='x', label='Test Points')
plt.scatter(X[y == 0, 0], X[y == 0, 1], c="r", marker="o", label="Class 0")
plt.scatter(X[y == 1, 0], X[y == 1, 1], c="b", marker="^", label="Class 1")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend(loc="upper left")
plt.title("1NN pred")
plt.show()
