# Classificando buscas utilizando biblioteca de analise de dados Pandas.

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


def fit_and_predict (nome ,modelo,treino_dados, treino_marcacoes, teste_dados,teste_marcacoes):

	modelo.fit(treino_dados, treino_marcacoes)

	resultado = modelo.predict(teste_dados)

	acertos = (resultado == teste_marcacoes)

	total_acertos = sum (acertos)
	total_elementos = len (teste_dados)
	taxa_acertos = 100.0 * total_acertos / total_elementos

	msg = "Taxa de acerto do algoritmo {0}: {1}".format(nome, taxa_acertos)
	print(msg)

	return taxa_acertos


from sklearn.naive_bayes import MultinomialNB
modelo_Multinomial = MultinomialNB ()

from sklearn.ensemble import AdaBoostClassifier
modelo_AdaBoost = AdaBoostClassifier ()

Resultado_Multinomial = fit_and_predict ("MultinomialNB", modelo_Multinomial, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

Resultado_AdaBoost = fit_and_predict ("AdaBoostClassifier", modelo_AdaBoost, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

# Teste do algoritimo base:
acertos_base = (Counter(teste_marcacoes).itervalues())
taxa_de_acerto_base = 100.0 * max(acertos_base)/ len (teste_marcacoes)
print ("A porcentagem do Algoritimo base: %f" % (taxa_de_acerto_base)) 

if (Resultado_Multinomial > Resultado_AdaBoost):
	vencedor = modelo_Multinomial
else:
	vencedor = modelo_AdaBoost

resultado = vencedor.predict(validacao_dados)

acertos = (resultado == validacao_marcacoes)

total_acertos = sum (acertos)
total_elementos = len (validacao_dados)
taxa_acertos = 100.0 * total_acertos / total_elementos

print ("Taxa de acerto do algoritmo Vencedor no mundo real: %f " % (taxa_acertos))
print ("Numero de dados da validacao: %d" % (total_elementos))