"""
Avalia a acuracia atual do Random Forest Obliquo
usando um split 80/20 do conjunto de treino.
"""
import numpy as np
import time
from random_forest import RandomForest

data = np.load("data.npz")
X_train = data["X_train"]
y_train = data["y_train"]

# Split 80/20 com seed fixa para reprodutibilidade
rng = np.random.default_rng(42)
n = len(X_train)
idx = rng.permutation(n)

split = int(0.8 * n)
X_tr, y_tr = X_train[idx[:split]], y_train[idx[:split]]
X_val, y_val = X_train[idx[split:]], y_train[idx[split:]]

print(f"Treino: {X_tr.shape[0]} amostras")
print(f"Validacao: {X_val.shape[0]} amostras")
print(f"Classes no val: {np.bincount(y_val)}")
print("-" * 40)

# Treinar e medir tempo
t0 = time.time()
forest = RandomForest(n_trees=100)
forest.fit(X_tr, y_tr)
t_train = time.time() - t0

# Predizer e medir acuracia
t0 = time.time()
y_pred = forest.predict(X_val)
t_pred = time.time() - t0

acc = np.mean(y_pred == y_val)

print(f"Acuracia: {acc:.4f} ({acc*100:.2f}%)")
print(f"Tempo de treino: {t_train:.1f}s")
print(f"Tempo de predicao: {t_pred:.1f}s")
print(f"Distribuicao predita: {np.bincount(y_pred)}")
print(f"Distribuicao real:    {np.bincount(y_val)}")
