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
x_mesh, y_mesh = np.meshgrid(np.arange(x_min, x_max + h, h), np.arange(y_min, y_max + h, h))
Z = np.array([predict(x, X, y) for x in np.c_[x_mesh.ravel(), y_mesh.ravel()]])
Z = Z.reshape(x_mesh.shape)
plt.pcolormesh(x_mesh, y_mesh, Z, cmap="Pastel1")
plt.scatter(X[y == 0, 0], X[y == 0, 1], c="b", marker="o", label="Class 0")
plt.scatter(X[y == 1, 0], X[y == 1, 1], c="r", marker="^", label="Class 1")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend(loc="upper left")
plt.title("1NN pred")
plt.show()
