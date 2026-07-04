import numpy as np
from sklearn.decomposition import PCA
import pandas as pd


# FASE 0 - CARREGAMENTO DE DADOS & DESCOBERTA
data = np.load("data.npz")

X_train = data["X_train"]
y_train = data["y_train"]
X_test = data["X_test"]

#print(X_train.shape) # LINHAS = AMOSTRAS | COLUNAS = FEATURES(X)
#print(y_train.shape) # AQUI OS Y DO X DE CIMA
#print(X_test.shape)

# INFOS SOBRE AS CLASSES
# No caso temos 3 classes, parecido com a prova [0,1,2] entao podemos no final classificar em diferentes formas
# Como se fosse animal marinho, terrestre e voador
#print(np.unique(y_train))
#print(len(np.unique(y_train)))

# NATUREZA DOS DADOS
# Ver se tem NaN, pensar no tipo pra escolher um PCA, SVM ou outor
# Ver ocorrencia de classes
#print(X_train.dtype)
#print(np.isnan(X_train).sum())

#print(X_train.min())
#print(X_train.max())

#print(np.bincount(y_train))

# FASE 1 - ANTES DA ARVORE TEM O NO
# Da entao pra comecar fazendo tipo uma funcao q pega e separa os dados
# Vai ser exatamente o que cada no vai fazer no final

# A ideia aqui eh simples achar o peso(w) de cada parametro(saber masi ou menos quem influencia mais na decisao)
# E tambem achar o tau, que determina onde via ser feito o corte
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

    # 1.ACHA OS PESOS
    pca = PCA(n_components=1)
    pca.fit(X_norm)

    w = pca.components_[0]

    z = X_norm@w # PROJECAO - Eh como se tivesse tres opntos distantes no espaco, mas ao olhar em uma direcao w
    # VOce eh capaz de ver quase como algo plano e pode cortar mais facil
    #print(z[:10])

    if np.min(z) == np.max(z): # Evita degeneracao
        return w, None, np.inf, None, None, -np.inf, mean, std, feature_indices

    # Gerou dados sem ordenacao eh tipo uma regua estranha, ent a ideia eh ordenar o y pensando nesse z
    # Da pra ver q o crote n eh bonitinho pra separar as classes ent usamos algo pra minimizar isso
    idx = np.argsort(z)

    z_sorted = z[idx]
    y_sorted = y[idx]

    #print(z_sorted[:20])
    #print(y_sorted[:20])

    # 2.ACHAR ONDE CORTAR
    # Vamos cacar os candidatos, a ideia eh testar pontos entre os achados pra n ficar infinitamente cacando
    best_tau = None
    best_gini = np.inf
    best_gain = np.inf

    # Identifica quais amostrar vao pra onde, pq no momento estamos no no mas precisa propagar pra arvore
    best_left_mask = None
    best_right_mask = None

    n = len(y)

    for i in range(n-1):
        if z_sorted[i] == z_sorted[i + 1]:
            continue
        tau = (z_sorted[i]+z_sorted[i+1])/2
        
        # Depois de achar o bendito, separamos, padrao
        left = y_sorted[:i + 1]
        right = y_sorted[i + 1:] # SUGESTAO BOA DO GPT, UM JEITO DIFERENTE DE PENSAR
        
        # Vemos se tem homogenidade entre eles, a pureza vista em aulas e atividades
        # Para facilitar separamos essa analise em um funcao
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

# FASE 2 - FAZER O NO VIRAR A ARVORE
class Node: # Eh o responsavel por guardar a estrtutura da arvore

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

    # Primeiro olhamos pra pureza pra saber se para ou continua a cirar no
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

    # Teste na raiz para analisar a proporcao
    #if depth == 0:
    #    print(np.bincount(y))
    #    print(np.bincount(y_left))
    #    print(np.bincount(y_right))
    
    if n_left == 0 or n_right == 0:
        node.prediction = np.bincount(y).argmax()
        return node

    if n_left == n or n_right == n:
        node.prediction = np.bincount(y).argmax()
        return node
    
    # Momento da recursao, eh onde vria nossa arvore
    node.w = w
    node.tau = tau
    node.mean = mean
    node.std = std
    node.feature_indices = feature_indices
    node.left = build_tree(X_left, y_left, depth + 1, max_depth, max_features, rng)
    node.right = build_tree(X_right, y_right, depth + 1, max_depth, max_features, rng)

    return node

# A arvore ta feita mas tem que caminhar por ela ne
def predict_one(node, x):
    # OLha pra folha
    if node.left is None and node.right is None:
        return node.prediction

    # se a folha n tiver sido bem definida
    if node.w is None or node.tau is None:
        return node.prediction

    # decisão de caminho (com normalização e sub-features locais)
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



#n = len(X_train)
#idx = np.random.permutation(n)

#train_idx = idx[:int(0.8*n)]
#val_idx = idx[int(0.8*n):]

#X_tr, y_tr = X_train[train_idx], y_train[train_idx]
#X_val, y_val = X_train[val_idx], y_train[val_idx]

#tree = build_tree(X_train, y_train, max_depth=6)
#y_pred = predict(tree, X_val)

#acc = np.mean(y_pred == y_val)
#print(acc)

# FASE 3 - RANDOM FOREST (BOOTSTRAP + VOTACAO)
class RandomForest:

    def __init__(self, n_trees, max_depth=20, max_features="sqrt", random_state=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_features = X.shape[1]

        # Calcula o número de features a amostrar por nó
        if self.max_features == "sqrt":
            mf = int(np.sqrt(n_features))
        elif self.max_features == "log2":
            mf = int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            mf = self.max_features
        else:
            mf = n_features  # Usa todas

        rng = np.random.default_rng(self.random_state)

        for i in range(self.n_trees):
            # Cada árvore recebe uma seed derivada para reprodutibilidade
            tree_rng = np.random.default_rng(rng.integers(0, 2**31))
            X_boot, y_boot = bootstrap_sample(X, y, tree_rng)
            tree = build_tree(X_boot, y_boot, max_depth=self.max_depth, max_features=mf, rng=tree_rng)
            self.trees.append(tree)

    def predict(self, X):
        all_predictions = []

        for tree in self.trees:
            pred = predict(tree,X)
            all_predictions.append(pred) # Cada linha via representar a predicao de cada arvores, cadaa mostra eh uma coluna

        all_predictions = np.array(all_predictions)
        all_predictions = all_predictions.T
        predictions = []

        for votes in all_predictions:
            winner = np.bincount(votes).argmax()
            predictions.append(winner)

        return np.array(predictions)

# O bootstrap gera um novo conjunto de treinamento para cada arvore, amostrando o dataset original com reposicao.
def bootstrap_sample(X, y, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n_samples = X.shape[0]
    indices_sorteados = rng.choice(n_samples, size=n_samples, replace=True)
    X_boot = X[indices_sorteados]
    y_boot = y[indices_sorteados]

    return X_boot, y_boot


# FASE 4 - AVALIAR O CONJUNTO DE TESTE
def main():
    forest = RandomForest(n_trees=100, max_depth=20)

    forest.fit(X_train, y_train)

    y_pred = forest.predict(X_test)

    ids = np.arange(1, len(y_pred) + 1)

    submission = pd.DataFrame({
        "ID": ids,
        "Prediction": y_pred
    })

    submission.to_csv("submission.csv", index=False)

    # CHeck de linhas
    #print(">>> LINHAS <<<")
    #print(len(y_pred))
    #print(X_test.shape[0])

    # Check das classes
    #print(">>> CLASSES <<<")
    #print(np.unique(y_pred))

    # Check distribuicao
    #print(">>> DISTRIBUICAO <<<")
    #print(np.bincount(y_pred))

    # Check Treino
    #print(">>> Treino <<<")
    #print(np.bincount(y_train))
    #tree = build_tree(X_train, y_train)

    #y_pred = predict(tree, X_train)
    #print(np.bincount(y_pred))

    #print(np.bincount(y_pred))
    #for tree in forest.trees:
    #    print(predict(tree, X_test[:1]))

if __name__ == "__main__":
    main()