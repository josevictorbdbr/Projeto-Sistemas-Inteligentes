import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from scipy.spatial.distance import cdist
import pickle
import math
import numpy as np

#Abrir arquivo e remover coluna DEATH_EVENT
dados = pd.read_csv('heart_failure_clinical_records_dataset.csv')
dados_pacientes = dados.drop(columns=['DEATH_EVENT'])

#Obter nomes das colunas
colunas = list(dados_pacientes.columns)

#Tratar valores 
preenchedor= SimpleImputer(strategy='median')
dados_preenchidos = preenchedor.fit_transform(dados_pacientes[colunas])
dados_preenchidos = pd.DataFrame(dados_preenchidos, columns=colunas)

#normalizar dados
normalizador = MinMaxScaler()

dados_pacientes_norm = normalizador.fit_transform(dados_preenchidos)
dados_pacientes_norm = pd.DataFrame(dados_pacientes_norm, columns=colunas)
print(dados_pacientes_norm)

#distorcoes
distorcoes = []
K = range(1, 11) 

for i in K:
    modelo_clusters = KMeans(n_clusters=i, random_state=42, n_init=10).fit(dados_pacientes_norm)
    distorcoes.append(
        sum(
            np.min(
                cdist(dados_pacientes_norm, modelo_clusters.cluster_centers_,'euclidean'), axis=1
                )/dados_pacientes_norm.shape[0]
            )
        )   
    
#Determinar numero otimo de clusters
x0 = K[0]
y0 = distorcoes[0]
xn = K[-1]
yn = distorcoes[-1]
distancias = []

for i in range(len(distorcoes)):
    x= K[i]
    y= distorcoes[i]
    numerador = abs((yn-y0)*x - (xn-x0)*y + xn*y0 - yn*x0)
    denominador = math.sqrt((yn-y0)**2 + (xn-x0)**2)
    distancias.append(numerador/denominador)
    
numero_clusters_otimo = K[distancias.index(np.max(distancias))]
print('Numero otimo de clusters =', numero_clusters_otimo)

#Treinar modelo final
cluster_heart  = KMeans(n_clusters=numero_clusters_otimo, random_state=42, n_init=10).fit(dados_pacientes_norm)

#salvar modelos pkl
modelo = {
    'preenchedor': preenchedor,
    'normalizador': normalizador,
    'cluster': cluster_heart
}
pickle.dump(modelo,open('modelos/modelo_heart.pkl', 'wb'))
pickle.dump(colunas,open('modelos/colunas_heart.pkl', 'wb'))

print('Treinamento concluido')