import numpy as np
from sklearn.decomposition import PCA
import pandas as pd

def split_node(X,y):
    pca = PCA(n_components=1)
    pca.fit(X)

    w = pca.components_[0]
    z = X@w 

    if np.min(z) == np.max(z):
        return w, None, np.inf, None, None, -np.inf

    idx = np.argsort(z)
    z_sorted = z[idx]
    y_sorted = y[idx]
    best_tau = None
    best_gini = np.inf
    best_gain = np.inf
    best_left_mask = None
    best_right_mask = None
    n = len(y)

    for i in range(n-1):
        if z_sorted[i] == z_sorted[i + 1]:
            continue

        tau = (z_sorted[i]+z_sorted[i+1])/2        
        left = y_sorted[:i + 1]
        right = y_sorted[i + 1:] 
        
        gini_total = (
            (len(left) / n) * gini(left)
            + (len(right) / n) * gini(right)
        )

        if gini_total < best_gini:
            best_gini = gini_total
            best_tau = tau
            mask = z <= tau
            best_left_mask = mask
            best_right_mask = ~mask

    gini_parent = gini(y)
    best_gain = gini_parent - best_gini 

    return w, best_tau, best_gini, best_left_mask, best_right_mask, best_gain

def gini(side):
    if len(side) == 0:
        return 0
    
    ocorrencia = np.bincount(side)
    num_element = len(side)
    prob = ocorrencia/num_element
    pureza = 1 - np.sum(prob**2)
    
    return pureza

class Node:

    def __init__(self):
        self.w = None
        self.tau = None
        self.left = None
        self.right = None
        self.prediction = None

def build_tree(X, y, depth=0, max_depth=15):
    node = Node()

    if depth >= max_depth:
        node.prediction = np.bincount(y).argmax()
        return node

    if len(np.unique(y)) == 1:
        node.prediction = y[0]
        return node
    if X.shape[0] == 1:
        node.prediction = y[0]
        return node


    w, tau, gini, best_left_mask, best_right_mask, gain = split_node(X, y)
    min_gain_threshold = 0.0
    if gain < min_gain_threshold:
        node.prediction = np.bincount(y).argmax()
        return node
    
    if tau is None:
        node.prediction = np.bincount(y).argmax()
        return node

    X_left = X[best_left_mask]
    y_left = y[best_left_mask]
    X_right = X[best_right_mask]
    y_right = y[best_right_mask]
    n = X.shape[0]
    n_left = X_left.shape[0]
    n_right = X_right.shape[0]

    if n_left == 0 or n_right == 0:
        node.prediction = np.bincount(y).argmax()
        return node

    if n_left == n or n_right == n:
        node.prediction = np.bincount(y).argmax()
        return node
    
    node.w = w
    node.tau = tau
    node.left = build_tree(X_left, y_left, depth + 1, max_depth)
    node.right = build_tree(X_right, y_right, depth + 1, max_depth)

    return node

def predict_one(node, x):

    if node.left is None and node.right is None:
        return node.prediction

    if node.w is None or node.tau is None:
        return node.prediction

    z = x @ node.w

    if z <= node.tau:
        return predict_one(node.left, x)
    else:
        return predict_one(node.right, x)


def predict(node, X):
    preds = []
    for i in range(X.shape[0]):
        preds.append(predict_one(node, X[i]))
    return np.array(preds)

class RandomForest:

    def __init__(self, n_trees):
        self.n_trees = n_trees
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            X_boot, y_boot = bootstrap_sample(X, y)
            tree = build_tree(X_boot, y_boot)
            self.trees.append(tree)

    def predict(self, X):
        all_predictions = []

        for tree in self.trees:
            pred = predict(tree,X)
            all_predictions.append(pred) 

        all_predictions = np.array(all_predictions)
        all_predictions = all_predictions.T
        predictions = []

        for votes in all_predictions:
            winner = np.bincount(votes).argmax()
            predictions.append(winner)

        return np.array(predictions)

def bootstrap_sample(X, y):
    rng = np.random.default_rng()
    n_samples = X.shape[0]
    indices_sorteados = rng.choice(n_samples, size=n_samples, replace=True)
    X_boot = X[indices_sorteados]
    y_boot = y[indices_sorteados]

    return X_boot, y_boot


def main():
    data = np.load("data.npz")

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]

    forest = RandomForest(n_trees=100)
    forest.fit(X_train, y_train)
    y_pred = forest.predict(X_test)
    ids = np.arange(1, len(y_pred) + 1)

    submission = pd.DataFrame({
        "ID": ids,
        "Prediction": y_pred
    })

    submission.to_csv("submission.csv", index=False)

if __name__ == "__main__":
    main()