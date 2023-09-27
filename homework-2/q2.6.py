#2.6
import numpy as np
import matplotlib.pyplot as plt

d1 = np.loadtxt("D1.txt")
d2 = np.loadtxt("D2.txt")

X_d1, y_d1 = d1[:, :-1], d1[:, -1]
X_d2, y_d2 = d2[:, :-1], d2[:, -1]

plt.figure(figsize=(6, 6))
plt.scatter(X_d1[y_d1 == 0, 0], X_d1[y_d1 == 0, 1], label="Class 0", c="blue", marker="o")
plt.scatter(X_d1[y_d1 == 1, 0], X_d1[y_d1 == 1, 1], label="Class 1", c="red", marker="x")
plt.title("D1")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(X_d2[y_d2 == 0, 0], X_d2[y_d2 == 0, 1], label="Class 0", c="blue", marker="o")
plt.scatter(X_d2[y_d2 == 1, 0], X_d2[y_d2 == 1, 1], label="Class 1", c="red", marker="x")
plt.title("D2")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()


#The decision tree is complex for D2 than D1 because there is a easy split that can happen in D1 because of the
#straight line with slope=0, which can easily represent something above Feature is class 0 and otherwise class 1.
#But if we look at D2 the line has some slope, at every Feature 2 there is a different Feature 1, causing the 
#tree to become complex. 