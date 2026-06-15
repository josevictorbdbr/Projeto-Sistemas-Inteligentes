import pandas as pd
import numpy as np
import pickle
from pprint import pprint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder

#Le o arquivo
dados = pd.read_csv('retail_black_friday_sales_100k.csv')

atributos = [
    'gender', 'city', 'customer_segment',
    'original_price', 'discount_pct', 'quantity',
    'purchase_hour', 'is_weekend', 'is_black_friday'
]

#Codificar atributos categoricos
encoders_features = {}
for coluna in ['gender', 'city', 'customer_segment']:
    le = LabelEncoder()
    dados[coluna] = le.fit_transform(dados[coluna])
    encoders_features[coluna] = le

dados_atributos = dados[atributos]
colunas = list(dados_atributos.columns)

rf_grid = {
    'n_estimators': [int(x) for x in np.linspace(start=10, stop=100, num=10)],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [2, 5, 10],
    'max_depth': [int(x) for x in np.linspace(start=10, stop=50, num=5)],
    'max_features': ['sqrt', 'log2']
}


def treinar_alvo(titulo_print, dados_atributos, dados_classe):
    print(f'\n---{titulo_print}---')

    encoder_alvo = LabelEncoder()
    classe_codificada = encoder_alvo.fit_transform(dados_classe)

    atributos_train, atributos_teste, \
        classe_train, classe_test = train_test_split(
            dados_atributos,
            classe_codificada,
            test_size=0.3,
            random_state=42,
            stratify=classe_codificada
        )

    scaler = StandardScaler()
    atributos_train_norm = scaler.fit_transform(atributos_train)
    atributos_teste_norm = scaler.transform(atributos_teste)

    rf = RandomForestClassifier(random_state=42)
    hyperparameters = RandomizedSearchCV(
        estimator=rf,
        param_distributions=rf_grid,
        n_iter=5,
        cv=3,
        verbose=0,
        n_jobs=1,
        random_state=42
    )
    hyperparameters.fit(atributos_train_norm, classe_train)

    print('Melhores parametros:')
    pprint(hyperparameters.best_params_)

    melhor_rf = hyperparameters.best_estimator_
    predicoes = melhor_rf.predict(atributos_teste_norm)

    acuracia = accuracy_score(classe_test, predicoes)
    print("Acuracia: ", acuracia)

    cm = confusion_matrix(classe_test, predicoes)
    print("\nMatriz de Confusão:")
    print(cm)

    #Acuracia, sensibilidade e especificidade por classe via matriz de confusão
    print("\nMetricas por Classe:")
    acuracia_classes = cm.diagonal() / cm.sum(axis=1)
    for i, (classe, acur_classe) in enumerate(zip(encoder_alvo.classes_, acuracia_classes)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sensibilidade  = tp / (tp + fn) if (tp + fn) > 0 else 0
        especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"  {classe}: acuracia={acur_classe:.4f}  sensibilidade={sensibilidade:.4f}  especificidade={especificidade:.4f}")

    f1 = f1_score(classe_test, predicoes, average='weighted')
    print("\nF1-Score:", f1)

    print("\nRelatório de Classificação:")
    print(classification_report(classe_test, predicoes, target_names=encoder_alvo.classes_, zero_division=0))

    return {
        'normalizador': scaler,
        'classificador': melhor_rf,
        'encoder_alvo': encoder_alvo,
        'acuracia': acuracia,
        'f1': f1
    }


modelo_categoria = treinar_alvo('PRODUCT CATEGORY', dados_atributos, dados['product_category'])
modelo_pagamento = treinar_alvo('PAYMENT METHOD', dados_atributos, dados['payment_method'])
modelo_faixa_etaria = treinar_alvo('AGE GROUP', dados_atributos, dados['age_group'])

print('\n---- COMPARACAO ----')
print(f'Product Category acuracia: {modelo_categoria["acuracia"]:.4f}  F1-Score: {modelo_categoria["f1"]:.4f}')
print(f'Payment Method acuracia: {modelo_pagamento["acuracia"]:.4f}  F1-Score: {modelo_pagamento["f1"]:.4f}')
print(f'Age Group acuracia: {modelo_faixa_etaria["acuracia"]:.4f}  F1-Score: {modelo_faixa_etaria["f1"]:.4f}')

resultados_finais = {
    'categoria': modelo_categoria,
    'pagamento': modelo_pagamento,
    'faixa_etaria': modelo_faixa_etaria,
    'colunas': colunas
}

pickle.dump(resultados_finais, open('modelos/modelos_bf.pkl', 'wb'))
pickle.dump(encoders_features, open('modelos/encoders_features.pkl', 'wb'))

print('\nModelos salvos')