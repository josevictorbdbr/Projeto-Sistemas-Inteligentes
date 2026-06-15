import pickle
import pandas as pd


#Carregar modelo e scaler
modelo = pickle.load(open('modelos/modelo_vinho.pkl', 'rb'))
colunas_vinho = pickle.load(open('modelos/colunas_vinho.pkl', 'rb'))

normalizador = modelo['normalizador']
classificador = modelo['classificador']

#Novo vinho
novo_vinho = pd.DataFrame([[
    7.4,      #Fixed acidity
    0.70,     #Volatile acidity
    0.00,     #Citric acid
    1.90,     #Residual sugar
    0.076,    #Chlorides
    11.0,     #Free sulfur dioxide
    34.0,     #Total sulfur dioxide
    0.9978,   #Densidade
    3.51,     #PH
    0.56,     #Sulphates
    9.4       #Alcohol
]], columns=colunas_vinho)


#Normalizar
novo_vinho_norm = pd.DataFrame(
    normalizador.transform(novo_vinho),
    columns=colunas_vinho
)

#Inferência
qualidade_prevista = classificador.predict(
    novo_vinho_norm
)

print(f'Qualidade prevista do vinho: {qualidade_prevista[0]}')