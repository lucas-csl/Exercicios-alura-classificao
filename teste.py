from collections import Counter
import pandas as pd

df = pd.read_csv ("buscas.csv")

X_df = df [["home", "busca", "logado"]]
Y_df = df ["comprou"]

X_dummies = pd.get_dummies(X_df)
Y_dummies = Y_df

X = X_dummies.values
Y = Y_dummies.values

porcentagem_treino = 0.8
porcentagem_teste = 0.1

tamanho_treino = int (round(len (Y) * porcentagem_treino))
tamanho_teste = int (round(len(Y) * porcentagem_teste) + tamanho_treino)
tamanho_validacao = len (Y) - tamanho_teste

treino_dados = X[:tamanho_treino]
treino_marcacoes = Y[:tamanho_treino]

teste_dados = X[tamanho_treino : tamanho_teste]
teste_marcacoes = Y[tamanho_treino : tamanho_teste]

validacao_dados = X[-tamanho_validacao:]
validacao_marcacoes = Y[-tamanho_validacao:]

print (X[799], treino_dados[-1:])
print (X[899], teste_dados[-1:])
print (X[999], validacao_dados[-1:])