import numpy as np
from sklearn.decomposition import PCA
import pandas as pd

def split_node(X, y, max_features=None, rng=None):
    n_features = X.shape[1]

    # Feature subsampling: seleciona sqrt(m) features aleatórias por nó
    if max_features is not None and max_features < n_features:
        feature_indices = rng.choice(n_features, size=max_features, replace=False)
        feature_indices.sort()
    else:
        feature_indices = np.arange(n_features)

    X_sub = X[:, feature_indices]

    # Normalização local: cada nó normaliza seus dados antes do PCA
    mean = np.mean(X_sub, axis=0)
    std = np.std(X_sub, axis=0)
    std[std == 0] = 1.0  # Evita divisão por zero em features constantes
    X_norm = (X_sub - mean) / std

    # Se todas as features selecionadas são constantes, não há como separar
    total_var = np.var(X_norm, axis=0).sum()
    if total_var == 0:
        w = np.zeros(len(feature_indices))
        return w, None, np.inf, None, None, -np.inf, mean, std, feature_indices

    pca = PCA(n_components=1)
    pca.fit(X_norm)

    w = pca.components_[0]
    z = X_norm @ w 

    if np.min(z) == np.max(z):
        return w, None, np.inf, None, None, -np.inf, mean, std, feature_indices

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

    return w, best_tau, best_gini, best_left_mask, best_right_mask, best_gain, mean, std, feature_indices

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
        self.mean = None            # Média local usada na normalização deste nó
        self.std = None             # Desvio padrão local usado na normalização deste nó
        self.feature_indices = None # Índices das features selecionadas neste nó
        self.left = None
        self.right = None
        self.prediction = None

def build_tree(X, y, depth=0, max_depth=15, max_features=None, rng=None):
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


    w, tau, gini, best_left_mask, best_right_mask, gain, mean, std, feature_indices = split_node(X, y, max_features, rng)
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
    node.mean = mean
    node.std = std
    node.feature_indices = feature_indices
    node.left = build_tree(X_left, y_left, depth + 1, max_depth, max_features, rng)
    node.right = build_tree(X_right, y_right, depth + 1, max_depth, max_features, rng)

    return node

def predict_one(node, x):

    if node.left is None and node.right is None:
        return node.prediction

    if node.w is None or node.tau is None:
        return node.prediction

    # Seleciona as mesmas features e aplica a mesma normalização do treino
    x_sub = x[node.feature_indices]
    x_norm = (x_sub - node.mean) / node.std
    z = x_norm @ node.w

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

    def __init__(self, n_trees, max_depth=15, max_features="sqrt", random_state=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_features = X.shape[1]

        if self.max_features == "sqrt":
            mf = int(np.sqrt(n_features))
        elif self.max_features == "log2":
            mf = int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            mf = self.max_features
        else:
            mf = n_features  

        rng = np.random.default_rng(self.random_state)

        for i in range(self.n_trees):

            tree_rng = np.random.default_rng(rng.integers(0, 2**31))
            X_boot, y_boot = bootstrap_sample(X, y, tree_rng)
            tree = build_tree(X_boot, y_boot, max_depth=self.max_depth, max_features=mf, rng=tree_rng)
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

def bootstrap_sample(X, y, rng=None):
    if rng is None:
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

    forest = RandomForest(n_trees=100, max_depth=20)
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