#2.7
import numpy as np
from collections import Counter

data = np.loadtxt("Dbig.txt")

np.random.seed(7)
np.random.shuffle(data)

train_size = 8192
train_set = data[:train_size]
test_set = data[train_size:]

class DecisionTree:
    def fit(self, X, y):
        self.tree = self.fitter(X, y)

    def fitter(self, X, y):
        samples, features = X.shape
        labels = len(np.unique(y))
        max_split = None
        max_gain = 0
        for feature in range(features):
            for threshold in np.unique(X[:, feature]):
                left = y[X[:, feature] < threshold]
                right = y[X[:, feature] >= threshold]
                if len(left) > 0 and len(right) > 0:
                    gain = self.infogain(y,left, right)
                    if gain > max_gain:
                        max_split = (feature, threshold)
                        max_gain = gain

        if max_gain == 0:
            return Counter(y).most_common(1)[0][0]

        feature, threshold = max_split
        left = X[:, feature] < threshold
        right = ~left
        l_tree = self.fitter(X[left], y[left])
        r_tree = self.fitter(X[right], y[right])
        return (feature, threshold, l_tree, r_tree)

    def entropy(self, y):
        _, l = np.unique(y, return_counts=True)
        prob = l / len(y)
        entropy = -np.sum(prob * np.log2(prob))
        return entropy

    def infogain(self, y, left, right):
        p = len(left) / len(y)
        q = len(right) / len(y)
        gain = self.entropy(y) - (p * self.entropy(left) + q * self.entropy(right))
        return gain

    def predict(self, X):
        return [self.predictor(x, self.tree) for x in X]

    def predictor(self, x, tree):
        if isinstance(tree, int) or isinstance(tree, float) or isinstance(tree, np.int64):
            return tree
        feature= tree[0]
        threshold= tree[1]
        left= tree[2] 
        right = tree[3]
        if x[feature] < threshold:
            return self.predictor(x, left)
        else:
            return self.predictor(x, right)
    def print_tree(self, node,features=None,classes=None, space=""):
        if isinstance(node, int) or isinstance(node,float):
            classname = classes[node] if classes else node
            print(space + "Predict", classes)
            return
        if features is None:
            feature = f"Feature {node[0]}"
        else:
            feature = features[node[0]]
        print(space + f"[{feature} < {node[1]}]")
        print(space + '--> True:')
        self.print_tree(node[2], features,classes, space+ "  ")
        print(space + '--> False:')
        self.print_tree(node[3], features,classes, space+ "  ")
    def node_count(self, node):
        if isinstance(node, int) or isinstance(node,float):
            return 1  # Leaf node
        else:
            feature, _, l, r = node
            left_count = self.node_count(l)
            right_count = self.node_count(r)
            return 1 + left_count + right_count
        
n_values = [32, 128, 512, 2048, 8192]
num_nodes = []
test_errors = [] 

def accuracy_score(y_true, y_pred):
    correct_predictions = 0
    total_samples = len(y_true)

    for true_label, predicted_label in zip(y_true, y_pred):
        if true_label == predicted_label:
            correct_predictions += 1

    accuracy = correct_predictions / total_samples

    return accuracy

for n in n_values:
    # Create the training set Dn
    train_data = train_set[:n]
    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    
    # Train a custom decision tree
    tree = DecisionTree()  # Customize this with your tree class
    tree.fit(X_train, y_train)
    
    # Calculate the number of nodes in the tree (you need to implement this)
    num_nodes.append(tree.node_count(tree.tree))  # Implement this method in your custom tree class
    
    # Predict on the test set and calculate the test set error (implement this)
    y_pred = tree.predict(test_set[:, :-1])  # Implement this method in your custom tree class
    test_error = 1 - accuracy_score(test_set[:, -1], y_pred)  # Implement accuracy_score
    test_errors.append(test_error)
    X_d1, y_d1 = train_data[:, :-1], train_data[:, -1]
    plt.figure(figsize=(8, 6))
    plt.scatter(X_d1[y_d1 == 0, 0], X_d1[y_d1 == 0, 1], label="Class 0", c="blue", marker="o")
    plt.scatter(X_d1[y_d1 == 1, 0], X_d1[y_d1 == 1, 1], label="Class 1", c="red", marker="x")
    plt.title("n="+str(n))
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()


print("Results:")
for i in range(len(n_values)):
    print(f"n = {n_values[i]}, Nodes = {num_nodes[i]}, Test Error = {test_errors[i]:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(num_nodes, test_errors, marker='o')
plt.title("Learning Curve")
plt.xlabel("Nodes")
plt.ylabel("Test Error")
plt.show()

