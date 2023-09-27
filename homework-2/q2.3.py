#2.3
import numpy as np
from collections import Counter

data = np.loadtxt("Druns.txt")

X = data[:, :-1]
y = data[:, -1]   

def entropy(y):
    _, n = np.unique(y, return_counts=True)
    prob = n / len(y)
    return -np.sum(prob * np.log2(prob + 1e-10))

H = entropy(y)

cuts = []
ig_ratios = []

for feature_i in range(X.shape[1]):
    feature_values = X[:, feature_i]
    unique_values = np.unique(feature_values)

    for threshold in unique_values:
        l = np.where(feature_values < threshold)
        r = np.where(feature_values >= threshold)
        H_left = entropy(y[l])
        H_right = entropy(y[r])

        Infogain = H - (len(l[0]) / len(y) * H_left + len(r[0]) / len(y) * H_right)

        Ig_ratio = Infogain / (entropy(feature_values)+ 1e-10)

        cuts.append((feature_i, threshold))
        ig_ratios.append(Ig_ratio)

        print(f"Cut: Feature {feature_i}, Threshold {threshold}")
        print(f"Information Gain: {Infogain}")
        print(f"Information Gain Ratio: {Info_gain_ratio}")
        print("\n")