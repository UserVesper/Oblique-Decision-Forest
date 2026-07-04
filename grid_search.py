"""
Grid Search completo para encontrar os melhores hiperparametros do Random Forest Obliquo.
Testa diferentes combinacoes de n_trees, max_depth, projection_method,
min_samples_split, min_samples_leaf e n_projections.
"""
import numpy as np
import time
from random_forest import RandomForest

# Carregar dados
data = np.load("data.npz")
X_train = data["X_train"]
y_train = data["y_train"]

# Split 80/20 com seed fixa para avaliacao justa
rng = np.random.default_rng(42)
n = len(X_train)
idx = rng.permutation(n)
split = int(0.8 * n)
X_tr, y_tr = X_train[idx[:split]], y_train[idx[:split]]
X_val, y_val = X_train[idx[split:]], y_train[idx[split:]]

print(f"Treino: {X_tr.shape[0]} | Validacao: {X_val.shape[0]}")
print("=" * 100)

# Defina a grade de parametros para testar.
# Sinta-se livre para adicionar/remover valores para ajustar o tempo de execucao.
param_grid = {
    "n_trees": [150, 160, 180],
    "max_depth": [18, 20, 23],
    "projection_method": ["lda"],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "n_projections": [3, 5],
}

# Gerando as combinacoes
import itertools
keys, values = zip(*param_grid.items())
experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

total = len(experiments)
print(f"Total de experimentos a rodar: {total}")
print("Pressione Ctrl+C a qualquer momento para parar. O ranking parcial sera salvo.")
print("=" * 100)

results = []
count = 0

try:
    for params in experiments:
        count += 1
        print(f"[{count}/{total}] "
              f"n_trees={params['n_trees']}, "
              f"depth={params['max_depth']}, "
              f"method={params['projection_method']}, "
              f"split={params['min_samples_split']}, "
              f"leaf={params['min_samples_leaf']}, "
              f"proj={params['n_projections']}", end=" ... ")

        t0 = time.time()
        forest = RandomForest(
            n_trees=params["n_trees"],
            max_depth=params["max_depth"],
            max_features="sqrt",
            projection_method=params["projection_method"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            n_projections=params["n_projections"],
            random_state=42
        )
        forest.fit(X_tr, y_tr)
        t_train = time.time() - t0

        y_pred = forest.predict(X_val)
        acc = np.mean(y_pred == y_val)

        print(f"acc={acc:.4f}  tempo={t_train:.1f}s")
        
        results.append({
            "params": params,
            "accuracy": acc,
            "train_time": t_train,
        })
except KeyboardInterrupt:
    print("\nExecucao interrompida pelo usuario. Mostrando resultados parciais...")

# Ordenar por acuracia (melhor primeiro)
results.sort(key=lambda x: x["accuracy"], reverse=True)

print("\n" + "=" * 100)
print(f"{'Rank':<5} {'Acuracia':<10} {'Tempo(s)':<10} {'Parametros':<70}")
print("-" * 100)
for i, r in enumerate(results[:30]):  # Mostrar top 30
    marker = " <-- MELHOR" if i == 0 else ""
    p = r["params"]
    p_str = f"n_trees={p['n_trees']} | depth={p['max_depth']} | method={p['projection_method']} | split={p['min_samples_split']} | leaf={p['min_samples_leaf']} | proj={p['n_projections']}"
    print(f"{i+1:<5} {r['accuracy']:<10.4f} {r['train_time']:<10.1f} {p_str}{marker}")

if results:
    best = results[0]
    p = best["params"]
    print(f"\nMelhor configuracao:")
    print(f"Acuracia: {best['accuracy']:.4f} ({best['accuracy']*100:.2f}%)")
    print(f"Parametros: {p}")
