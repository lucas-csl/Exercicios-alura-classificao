from dados import carregar_acessos

X, Y = carregar_acessos ()

treino_dados = X[:90]
treino_marcacao = Y[:90]

teste_dados = X[-9:]
teste_marcacao = Y[-9:]

from sklearn.naive_bayes import MultinomialNB

modelo = MultinomialNB ()
modelo.fit (treino_dados, treino_marcacao)

resultado = modelo.predict(teste_dados)

diferenca = resultado - teste_marcacao

acertos = [d for d in diferenca if d == 0]
total_acertos = len (acertos)
total_elementos = len (teste_dados)
taxa_acertos = 100.0 * total_acertos / total_elementos

print (taxa_acertos)
print (total_elementos)
