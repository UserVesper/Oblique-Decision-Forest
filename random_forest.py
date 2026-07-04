import numpy as np
import warnings
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd

def split_node(X, y, max_features=None, rng=None, projection_method="lda", min_samples_leaf=1):
    n_features = X.shape[1]

    if max_features is not None and max_features < n_features:
        feature_indices = rng.choice(n_features, size=max_features, replace=False)
        feature_indices.sort()
    else:
        feature_indices = np.arange(n_features)

    X_sub = X[:, feature_indices]

    mean = np.mean(X_sub, axis=0)
    std = np.std(X_sub, axis=0)
    std[std == 0] = 1.0
    X_norm = (X_sub - mean) / std

    total_var = np.var(X_norm, axis=0).sum()
    if total_var == 0:
        w = np.zeros(len(feature_indices))
        return w, None, np.inf, None, None, -np.inf, mean, std, feature_indices

    use_pca = True
    if projection_method == "lda" and len(np.unique(y)) >= 2:
        try:
            lda = LDA(n_components=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                lda.fit(X_norm, y)
            w = lda.scalings_[:, 0]
            norm = np.linalg.norm(w)
            if norm > 0:
                w = w / norm
                use_pca = False
        except Exception:
            use_pca = True

    if use_pca:
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
    best_left_mask = None
    best_right_mask = None
    n = len(y)

    n_classes = int(y.max()) + 1

    # Contadores incrementais de Gini
    left_counts = np.zeros(n_classes, dtype=int)
    right_counts = np.bincount(y_sorted, minlength=n_classes).astype(int)
    n_left = 0
    n_right = n

    for i in range(n - 1):
        c = y_sorted[i]
        left_counts[c] += 1
        right_counts[c] -= 1
        n_left += 1
        n_right -= 1

        if z_sorted[i] == z_sorted[i + 1]:
            continue

        # Respeita min_samples_leaf
        if n_left < min_samples_leaf or n_right < min_samples_leaf:
            continue

        tau = (z_sorted[i] + z_sorted[i + 1]) / 2        

        gini_left = 1.0 - np.sum((left_counts / n_left) ** 2)
        gini_right = 1.0 - np.sum((right_counts / n_right) ** 2)
        gini_total = (n_left / n) * gini_left + (n_right / n) * gini_right

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
        self.mean = None            
        self.std = None             
        self.feature_indices = None 
        self.left = None
        self.right = None
        self.prediction = None
        self.class_proba = None

def build_tree(X, y, depth=0, max_depth=20, max_features=None, rng=None,
               projection_method="lda", min_samples_split=5, min_samples_leaf=2,
               n_projections=3, n_classes=None):
    node = Node()

    if n_classes is None:
        n_classes = int(y.max()) + 1

    counts = np.bincount(y, minlength=n_classes)

    if depth >= max_depth or len(np.unique(y)) == 1 or X.shape[0] < min_samples_split:
        node.prediction = counts.argmax()
        node.class_proba = counts / counts.sum()
        return node

    # Tenta multiplas projecoes e escolhe o melhor split
    best_result = None
    best_split_gain = -np.inf

    for _ in range(n_projections):
        result = split_node(X, y, max_features, rng, projection_method, min_samples_leaf)
        w, tau, gini_val, left_mask, right_mask, gain, mean, std, feat_idx = result
        if tau is not None and gain > best_split_gain:
            best_split_gain = gain
            best_result = result

    if best_result is None or best_split_gain <= 0.0:
        node.prediction = counts.argmax()
        node.class_proba = counts / counts.sum()
        return node

    w, tau, _, best_left_mask, best_right_mask, _, mean, std, feature_indices = best_result

    X_left = X[best_left_mask]
    y_left = y[best_left_mask]
    X_right = X[best_right_mask]
    y_right = y[best_right_mask]
    n = X.shape[0]
    n_left = X_left.shape[0]
    n_right = X_right.shape[0]

    if n_left == 0 or n_right == 0 or n_left == n or n_right == n:
        node.prediction = counts.argmax()
        node.class_proba = counts / counts.sum()
        return node
    
    node.w = w
    node.tau = tau
    node.mean = mean
    node.std = std
    node.feature_indices = feature_indices
    node.left = build_tree(X_left, y_left, depth + 1, max_depth, max_features, rng,
                           projection_method, min_samples_split, min_samples_leaf,
                           n_projections, n_classes)
    node.right = build_tree(X_right, y_right, depth + 1, max_depth, max_features, rng,
                            projection_method, min_samples_split, min_samples_leaf,
                            n_projections, n_classes)

    return node

def predict_one(node, x):

    if node.left is None and node.right is None:
        return node.prediction

    if node.w is None or node.tau is None:
        return node.prediction

    x_sub = x[node.feature_indices]
    x_norm = (x_sub - node.mean) / node.std
    z = x_norm @ node.w

    if z <= node.tau:
        return predict_one(node.left, x)
    else:
        return predict_one(node.right, x)

def predict_proba_one(node, x):
    if node.left is None and node.right is None:
        return node.class_proba

    if node.w is None or node.tau is None:
        return node.class_proba

    x_sub = x[node.feature_indices]
    x_norm = (x_sub - node.mean) / node.std
    z = x_norm @ node.w

    if z <= node.tau:
        return predict_proba_one(node.left, x)
    else:
        return predict_proba_one(node.right, x)


def predict(node, X):
    preds = []
    for i in range(X.shape[0]):
        preds.append(predict_one(node, X[i]))
    return np.array(preds)

class RandomForest:

    def __init__(self, n_trees, max_depth=20, max_features="sqrt",
                 projection_method="lda", min_samples_split=5,
                 min_samples_leaf=2, n_projections=3, random_state=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.projection_method = projection_method
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_projections = n_projections
        self.random_state = random_state
        self.trees = []
        self.n_classes_ = None

    def fit(self, X, y):
        self.trees = []
        self.n_classes_ = int(y.max()) + 1
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
            tree = build_tree(X_boot, y_boot, max_depth=self.max_depth,
                              max_features=mf, rng=tree_rng,
                              projection_method=self.projection_method,
                              min_samples_split=self.min_samples_split,
                              min_samples_leaf=self.min_samples_leaf,
                              n_projections=self.n_projections,
                              n_classes=self.n_classes_)
            self.trees.append(tree)

    def predict(self, X):
        # Soft voting: soma as probabilidades de cada árvore
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X):
        all_proba = np.zeros((X.shape[0], self.n_classes_))

        for tree in self.trees:
            for i in range(X.shape[0]):
                all_proba[i] += predict_proba_one(tree, X[i])

        all_proba /= len(self.trees)
        return all_proba

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

    forest = RandomForest(
        n_trees=150,
        max_depth=18,
        max_features="sqrt",
        projection_method="lda",
        min_samples_split=5,
        min_samples_leaf=2,
        n_projections=5,
        random_state=42,
    )
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