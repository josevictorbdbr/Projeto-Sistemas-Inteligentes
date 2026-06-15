import pickle
import pandas as pd

#Carregar modelo e encoders
modelos = pickle.load(open('modelos/modelos_bf.pkl', 'rb'))
encoders_features = pickle.load(open('modelos/encoders_features.pkl', 'rb'))

colunas = modelos['colunas']

#Dados da venda a classificar
nova_venda = {
    'gender': 'Male',
    'city': 'New York',
    'customer_segment': 'New',
    'original_price': 199.99,
    'discount_pct': 20,
    'quantity': 2,
    'purchase_hour': 14,
    'is_weekend': 1,
    'is_black_friday': 1
}

df_venda = pd.DataFrame([nova_venda])

#Aplicar os mesmos encoders usados no treino
for col in ['gender', 'city', 'customer_segment']:
    df_venda[col] = encoders_features[col].transform(df_venda[col])

#garantir ordem
df_venda = df_venda[colunas]

def inferir(modelo_dict, df):
    normalizador  = modelo_dict['normalizador']
    classificador = modelo_dict['classificador']
    encoder_alvo  = modelo_dict['encoder_alvo']

    #Normalizar
    df_norm    = normalizador.transform(df)
    
    #Inferência
    classe_idx = classificador.predict(df_norm)[0]
    classe = encoder_alvo.inverse_transform([classe_idx])[0]
    score = classificador.predict_proba(df_norm)[0][classe_idx]

    return classe, score

#Inferência dos alvos
cat, score_cat = inferir(modelos['categoria'], df_venda)
pag, score_pag = inferir(modelos['pagamento'], df_venda)
age, score_age = inferir(modelos['faixa_etaria'], df_venda)

print(f'Categoria do produto prevista: {cat} (Certeza: {score_cat*100:.2f}%)')
print(f'Forma de pagamento prevista: {pag} (Certeza: {score_pag*100:.2f}%)')
print(f'Faixa etária prevista: {age} (Certeza: {score_age*100:.2f}%)')