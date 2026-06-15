import pickle
import pandas as pd


#abrir dados e tirar o DEATH_EVENT
dados = pd.read_csv('heart_failure_clinical_records_dataset.csv')
dados_pacientes = dados.drop(columns=['DEATH_EVENT'])

#Carregar arquivos após treinamento
modelo = pickle.load(open('modelos/modelo_heart.pkl', 'rb'))
colunas_heart = pickle.load(open('modelos/colunas_heart.pkl', 'rb'))

#Recuperar
preenchedor = modelo['preenchedor']
normalizador = modelo['normalizador']
cluster_heart = modelo['cluster']

#garantir ordem
dados_pacientes = dados_pacientes[colunas_heart]

#preencher valores que faltam
dados_pacientes_preenchidos = preenchedor.transform(dados_pacientes)
dados_pacientes_preenchidos = pd.DataFrame(dados_pacientes_preenchidos, columns=colunas_heart)

#Normalizar
dados_pacientes_norm = normalizador.transform(dados_pacientes_preenchidos)    
dados_pacientes_norm = pd.DataFrame(dados_pacientes_norm, columns=colunas_heart)
    
dados['cluster'] = cluster_heart.predict(dados_pacientes_norm)

#Mostrar quantidade de pacientes por cluster
print('\nQuantidade de pacientes por cluster:\n')
print(dados['cluster'].value_counts().sort_index().to_string())


#Mostrar media das variaveis por cluster
resumo_clusters = dados.drop(columns=['DEATH_EVENT'], errors='ignore').groupby('cluster').mean(numeric_only=True)
print(resumo_clusters)

print("\nCentros dos clusters:\n")
print(pd.DataFrame(cluster_heart.cluster_centers_,columns=colunas_heart))