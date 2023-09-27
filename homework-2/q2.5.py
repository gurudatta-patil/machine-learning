#2.5
import numpy as np
from collections import Counter

data = np.loadtxt("D1.txt")
X = data[:, :-1]  
y = data[:, -1]  
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
            print(space + "Predict", classname)
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
        
X_train = X
y_train = y

tree = DecisionTree()
tree.fit(X_train, y_train)
feature_names=["Feature 1", "Feature 2"]
print("Tree for D!")
tree.print_tree(tree.tree, feature_names)

data = np.loadtxt("D2.txt")
X = data[:, :-1]  
y = data[:, -1]  

X_train = X
y_train = y
print("\n\n")
tree = DecisionTree()
tree.fit(X_train, y_train)
feature_names=["Feature 1", "Feature 2"]
print("Tree for D2")
tree.print_tree(tree.tree, feature_names)