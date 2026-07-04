"""
Grid Search para encontrar os melhores hiperparametros do Random Forest Obliquo.
Testa diferentes combinacoes de n_trees e max_depth.
"""
import numpy as np
import time
from random_forest import RandomForest

data = np.load("data.npz")
X_train = data["X_train"]
y_train = data["y_train"]

# Split 80/20 com seed fixa
rng = np.random.default_rng(42)
n = len(X_train)
idx = rng.permutation(n)
split = int(0.8 * n)
X_tr, y_tr = X_train[idx[:split]], y_train[idx[:split]]
X_val, y_val = X_train[idx[split:]], y_train[idx[split:]]

print(f"Treino: {X_tr.shape[0]} | Validacao: {X_val.shape[0]}")
print("=" * 65)

# Parametros a testar
param_grid = {
    "n_trees": [50, 100, 150, 200],
    "max_depth": [5, 10, 15, 20, 25],
}

results = []

total = len(param_grid["n_trees"]) * len(param_grid["max_depth"])
count = 0

for n_trees in param_grid["n_trees"]:
    for max_depth in param_grid["max_depth"]:
        count += 1
        print(f"[{count}/{total}] n_trees={n_trees}, max_depth={max_depth}", end=" ... ")

        t0 = time.time()
        forest = RandomForest(
            n_trees=n_trees,
            max_depth=max_depth,
            max_features="sqrt",
            random_state=42,
        )
        forest.fit(X_tr, y_tr)
        t_train = time.time() - t0

        y_pred = forest.predict(X_val)
        acc = np.mean(y_pred == y_val)

        print(f"acc={acc:.4f}  tempo={t_train:.1f}s")
        results.append({
            "n_trees": n_trees,
            "max_depth": max_depth,
            "accuracy": acc,
            "train_time": t_train,
        })

# Ordenar por acuracia (melhor primeiro)
results.sort(key=lambda x: x["accuracy"], reverse=True)

print("\n" + "=" * 65)
print(f"{'Rank':<5} {'n_trees':<10} {'max_depth':<12} {'Acuracia':<12} {'Tempo(s)':<10}")
print("-" * 65)
for i, r in enumerate(results):
    marker = " <-- MELHOR" if i == 0 else ""
    print(f"{i+1:<5} {r['n_trees']:<10} {r['max_depth']:<12} {r['accuracy']:<12.4f} {r['train_time']:<10.1f}{marker}")

best = results[0]
print(f"\nMelhor configuracao: n_trees={best['n_trees']}, max_depth={best['max_depth']}")
print(f"Acuracia: {best['accuracy']:.4f} ({best['accuracy']*100:.2f}%)")
