# Avaliação Semestral — Sistemas Inteligentes

Projeto com três problemas de Machine Learning: clustering, classificação e inferência.

## Como Rodar:
- Clone o repositório e entre na pasta do projeto com `cd`
- Rode o treinamento do problema desejado (ex: `python wine_treinamento.py`)
- Rode o inferir ou descrever (ex: `python wine_inferir.py`)
- Os arquivos `.pkl` são gerados localmente na pasta `modelos/`

## 1. Heart Failure (Clustering)

**Objetivo:** Agrupar pacientes e identificar o cluster de um novo paciente.

**Dataset:** https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records

### Pipeline
- Remoção da coluna `DEATH_EVENT`
- Tratamento de valores ausentes com mediana (`SimpleImputer`)
- Normalização com `MinMaxScaler`
- Definição do número de clusters (método do cotovelo)
- Treinamento com `KMeans`

**Resultado:** Agrupamento de pacientes baseado em características clínicas.


## 2. Wine Quality (Classificação)

**Objetivo:** Prever a qualidade do vinho (tinto e branco).

**Dataset:** https://archive.ics.uci.edu/dataset/186/wine+quality

### Pipeline
- Leitura dos datasets (tinto e branco)
- União dos dados em um único DataFrame
- Separação de features e target (`quality`)
- Split 70/30 com estratificação
- Balanceamento com `SMOTE` (somente treino)
- Normalização com `StandardScaler`
- Treinamento dos modelos:
  - Random Forest
  - SVM
  - Logistic Regression
- Otimização com `RandomizedSearchCV`
- Seleção do melhor modelo via F1-score (weighted)

**Resultado:** Random Forest foi o melhor modelo.


## 3. Black Friday Sales (Classificação múltipla)

**Objetivo:** Prever:
- Categoria do produto
- Forma de pagamento
- Faixa etária do comprador

**Dataset:** https://www.kaggle.com/datasets/noopurbhatt/retail-black-friday-sales-dataset

### Pipeline
- Codificação de variáveis categóricas com `LabelEncoder`
- Split 70/30 estratificado
- Normalização com `StandardScaler`
- Treinamento de 3 modelos Random Forest (um por alvo)
- Otimização com `RandomizedSearchCV`
- Avaliação com métricas: especificidade e sensibilidade
- Uso de `predict_proba` para confiança das previsões
