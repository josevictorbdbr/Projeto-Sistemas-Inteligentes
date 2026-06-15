import pandas as pd
import numpy as np
import pickle
from pprint import pprint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE


#Abrir arquivos de dados e juntar ambos
vinho_tinto = pd.read_csv('winequality-red.csv', sep=';')
vinho_branco = pd.read_csv('winequality-white.csv', sep=';')

dados = pd.concat([vinho_tinto, vinho_branco],ignore_index=True)


#Separar atributos e classe
dados_atributos = dados.drop(columns=['quality'])
colunas = list(dados_atributos.columns)
dados_classe = dados['quality']


#segmentar os dados em dados para treinamento e dados para teste
atributos_train, atributos_teste, \
    classe_train,classe_test = train_test_split(
            dados_atributos,
            dados_classe,
            test_size=0.3,
            random_state=42,
            stratify=dados_classe
        )

#Normalizar
smote = SMOTE(random_state=42, k_neighbors=2)

atributos_train, classe_train = smote.fit_resample(
    atributos_train,
    classe_train
)

scaler = StandardScaler()
atributos_train = scaler.fit_transform(atributos_train)
atributos_teste = scaler.transform(atributos_teste)


#-----------------
#--RANDOM FOREST--
#-----------------

print("--RANDOM FOREST--")

n_estimators = [int(x) for x in np.linspace(start=10, stop=100, num=10)]
criterion = ['gini', 'entropy']
min_samples_split = [int(x) for x in np.linspace(start=2, stop=10, num=2)]
max_depth = [int(x) for x in np.linspace(start=10, stop=100, num=20)]
max_features = ['sqrt', 'log2']

#criar a grade de valores RF
rf_grid={
    'n_estimators': n_estimators,
    'criterion': criterion,
    'min_samples_split':min_samples_split,
    'max_depth': max_depth, 
    'max_features': max_features
}

rf = RandomForestClassifier(random_state=42)
rf_hyperparameters = RandomizedSearchCV(
    estimator=rf,
    param_distributions=rf_grid,
    n_iter=5,
    cv=3,
    verbose=0,
    n_jobs=1,
    random_state=42
)
rf_hyperparameters.fit(atributos_train, classe_train)

#Mostrar o resultado da hiperparametrização RF
print('Melhores parametros:')
pprint(rf_hyperparameters.best_params_)

#Recuperar melhor modelo RF
melhor_rf = rf_hyperparameters.best_estimator_
predicoes_rf = melhor_rf.predict(atributos_teste)

#Calcular Acuracia RF
acuracia_rf = accuracy_score(
    classe_test,
    predicoes_rf
)
print("Acuracia: ", acuracia_rf)

#Matriz de confusão RF
cm_rf = confusion_matrix(classe_test,predicoes_rf)
print("\nMatriz de Confusão:")
print(cm_rf)

#Acuracia das classes RF
print("\nAcuracia por Classe:")
acuracia_classes_rf = \
    cm_rf.diagonal() / cm_rf.sum(axis=1)

for classe, acuracia in zip(
    melhor_rf.classes_,
    acuracia_classes_rf
):  print(f"Classe {classe}: {acuracia:.4f}")
    
#
f1_rf = f1_score(
    classe_test,
    predicoes_rf,
    average='weighted'
)
print("\nF1-Score:", f1_rf)

print("\nRelatório de Classificação:")
print(classification_report(
    classe_test,
    predicoes_rf,
    zero_division=0
))


#-------------------
#--------SVM--------
#-------------------

print("--------SVM--------")

svm_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf'],
    'gamma': ['scale']
}

svm = SVC(random_state=42)

svm_hyperparameters = RandomizedSearchCV(
    estimator=svm,
    param_distributions=svm_grid,
    n_iter=3,
    cv=3,
    verbose=0,
    n_jobs=1,
    random_state=42
)
svm_hyperparameters.fit(atributos_train, classe_train)

#Mostrar o resultado da hiperparametrização SVM
print('Melhores parametros:')
pprint(svm_hyperparameters.best_params_)

melhor_svm = svm_hyperparameters.best_estimator_
predicoes_svm = melhor_svm.predict(atributos_teste)

#Calcular Acuracia SVM
acuracia_svm = accuracy_score(classe_test,predicoes_svm)
print('Acuracia:', acuracia_svm)

#Matriz de confusão SVM
cm_svm = confusion_matrix(classe_test,predicoes_svm)
print("\nMatriz de Confusão:")
print(cm_svm)

#Acuracia das classes SVM
print("\nAcuracia por Classe:")
acuracia_classes_svm = \
    cm_svm.diagonal() / cm_svm.sum(axis=1)

for classe, acuracia in zip(
    melhor_svm.classes_,
    acuracia_classes_svm
):  print(f"Classe {classe}: {acuracia:.4f}")

f1_svm = f1_score(
    classe_test,
    predicoes_svm,
    average='weighted'
)
print("\nF1-Score:", f1_svm)

print(classification_report(
    classe_test,
    predicoes_svm,
    zero_division=0
))


#-----------------------
#--LOGISTIC REGRESSION--
#-----------------------

print("--LOGISTIC REGRESSION--")

lr_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs']
}
lr = LogisticRegression(max_iter=1000, random_state=42)

lr_hyperparameters = RandomizedSearchCV(
    estimator=lr,
    param_distributions=lr_grid,
    n_iter=5,
    cv=3,
    verbose=0,
    n_jobs=1,
    random_state=42
)

lr_hyperparameters.fit(atributos_train, classe_train)

#Mostrar o resultado da hiperparametrização LR
print('Melhores parametros:')
pprint(lr_hyperparameters.best_params_)

melhor_lr = lr_hyperparameters.best_estimator_
predicoes_lr = melhor_lr.predict(atributos_teste)

#Calcular Acuracia LR
acuracia_lr = accuracy_score(classe_test,predicoes_lr)
print('Acuracia:', acuracia_lr)

#Matriz de confusão LR
cm_lr = confusion_matrix(classe_test,predicoes_lr)

print("\nMatriz de Confusao:")
print(cm_lr)

print("\nAcuracia por Classe:")
acuracia_classes_lr = \
    cm_lr.diagonal() / cm_lr.sum(axis=1)

for classe, acuracia in zip(
    melhor_lr.classes_,
    acuracia_classes_lr
):  print(f"Classe {classe}: {acuracia:.4f}")

f1_lr = f1_score(
    classe_test,
    predicoes_lr,
    average='weighted'
)
print("\nF1-Score:", f1_lr)

print(classification_report(
    classe_test,
    predicoes_lr,
    zero_division=0
))


#----Comparacao Final----
print('\n---- COMPARACAO FINAL ----')

print(f'Random Forest')
print(f'  Acurácia: {acuracia_rf:.4f}')
print(f'  F1-Score: {f1_rf:.4f}')

print(f'\nSVM')
print(f'  Acurácia: {acuracia_svm:.4f}')
print(f'  F1-Score: {f1_svm:.4f}')

print(f'\nLogistic Regression')
print(f'  Acurácia: {acuracia_lr:.4f}')
print(f'  F1-Score: {f1_lr:.4f}')


#Salva dps de saber o melhor modelo entre os 3
resultados = {
    'Random Forest': f1_rf,
    'SVM': f1_svm,
    'Logistic Regression': f1_lr
}

melhor_modelo = max(resultados,key=resultados.get)

print('\nMELHOR MODELO ENTRE OS 3:')
print(melhor_modelo)
print('F1-Score:', resultados[melhor_modelo])

if melhor_modelo == 'Random Forest':
    modelo_final = melhor_rf
elif melhor_modelo == 'SVM':
    modelo_final = melhor_svm
else:
    modelo_final = melhor_lr

modelo = {
    'normalizador': scaler,
    'classificador': modelo_final
}

pickle.dump(modelo,open('modelos/modelo_vinho.pkl', 'wb'))
pickle.dump(colunas,open('modelos/colunas_vinho.pkl', 'wb'))

print('\nModelo salvo')