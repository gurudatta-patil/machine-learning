#3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = np.loadtxt("Dbig.txt")

np.random.seed(7)
np.random.shuffle(data)

n_nodes = []
t_errors = [] 

n_values = [32, 128, 512, 2048, 8192]

for n in n_values:
    train_data = data[:n]
    X_train, y_train = train_data[:, :-1], train_data[:, -1]

    tree = DecisionTreeClassifier(random_state=7)
    tree.fit(X_train, y_train)

    n_nodes.append(tree.tree_.node_count)

    X_test, y_test = data[n:, :-1], data[n:, -1]
    y_pred = tree.predict(X_test)
    t_error = 1 - accuracy_score(y_test, y_pred)
    t_errors.append(t_error)

print("Results:")
for i in range(len(n_values)):
    print(f"n = {n_values[i]}, Nodes = {n_nodes[i]}, Test Error = {t_errors[i]:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(n_nodes, t_errors, marker='o')
plt.title("Learning Curve")
plt.xlabel("Nodes")
plt.ylabel("Test Error")
plt.show()
