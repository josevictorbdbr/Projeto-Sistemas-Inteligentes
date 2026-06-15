import pickle
import pandas as pd


#Carregar modelos
modelo = pickle.load(open('modelos/modelo_heart.pkl', 'rb'))
colunas_heart = pickle.load(open('modelos/colunas_heart.pkl', 'rb'))

preenchedor = modelo['preenchedor']
normalizador = modelo['normalizador']
cluster_heart = modelo['cluster']

#Paciente novo
novo_paciente = pd.DataFrame([[
    65,         #Idade
    1,          #Anaemia
    582,        #Creatinine_phosphokinase
    0,          #Diabetes
    38,         #Ejection_fraction
    1,          #Pressao alta
    263358,     #Platelets
    1.2,        #Serum_creatinine
    136,        #Serum_sodium
    1,          #Sexo
    0,          #Fuma
    120         #Tempo
]], columns=colunas_heart)

#Aplicar preenchimento e normalizacao
novo_paciente_preenchido = pd.DataFrame(preenchedor.transform(novo_paciente),columns=colunas_heart)
novo_paciente_norm = pd.DataFrame(normalizador.transform(novo_paciente_preenchido),columns=colunas_heart)

#Inferência
cluster_paciente = cluster_heart.predict(novo_paciente_norm)
print(f'Paciente pertence ao Cluster {cluster_paciente[0]}')