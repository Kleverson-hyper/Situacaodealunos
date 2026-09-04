# ============================================================
# Preditor de Situação do Aluno — versão Streamlit
# ============================================================
# Mesma lógica de dados e modelo da versão anterior (Gradio),
# agora com a interface reescrita em Streamlit.
# Para rodar:  streamlit run app.py
# ============================================================

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------------------------------------
# Configuração da página
# ------------------------------------------------------------
st.set_page_config(page_title="Preditor de Situação do Aluno", page_icon="🎓", layout="centered")

# ------------------------------------------------------------
# 1. Base de dados (cacheada para não recriar a cada interação)
# ------------------------------------------------------------
@st.cache_data
def carregar_dados():
    np.random.seed(42)
    dados = {
        'Horas_de_estudo': [
            10, 2, 5, 8, 1, 9, 3, 7, 6, 4,
            12, 1, 6, 8, 2, 5, 9, 3, 10, 7,
            4, 6, 2, 8, 5, 9, 1, 7, 3, 10,
            6, 4, 8, 2, 9, 5, 7, 1, 10, 3,
            6, 8, 4, 9, 2, 7, 5, 10, 1, 6,
            3, 8, 5, 9, 2, 7, 4, 10, 6, 1
        ],
        'Faltas': [
            2, 15, 6, 1, 20, 0, 12, 3, 5, 9,
            0, 18, 4, 2, 16, 7, 1, 14, 0, 3,
            10, 5, 17, 2, 8, 1, 19, 3, 13, 0,
            4, 9, 1, 15, 0, 6, 2, 20, 0, 12,
            5, 2, 10, 1, 16, 3, 7, 0, 19, 4,
            13, 2, 6, 1, 17, 3, 9, 0, 5, 20
        ],
        'Nota': [
            8.5, 3.0, 6.5, 9.0, 2.5, 9.5, 4.0, 8.0, 7.0, 5.5,
            9.8, 2.0, 6.8, 9.2, 3.2, 6.0, 9.0, 4.5, 9.9, 7.8,
            5.0, 7.2, 3.5, 8.8, 6.2, 9.3, 2.2, 8.1, 4.2, 9.7,
            7.0, 5.2, 8.6, 3.0, 9.4, 6.4, 7.9, 2.1, 9.6, 4.8,
            6.9, 8.7, 5.4, 9.1, 3.4, 8.0, 6.6, 9.8, 2.0, 7.1,
            4.6, 8.9, 6.3, 9.0, 3.1, 7.6, 5.6, 9.9, 6.7, 2.3
        ],
        'Situacao': [
            'Aprovado','Reprovado','Recuperação','Aprovado','Reprovado',
            'Aprovado','Reprovado','Aprovado','Recuperação','Recuperação',
            'Aprovado','Reprovado','Recuperação','Aprovado','Reprovado',
            'Recuperação','Aprovado','Reprovado','Aprovado','Aprovado',
            'Recuperação','Recuperação','Reprovado','Aprovado','Recuperação',
            'Aprovado','Reprovado','Aprovado','Reprovado','Aprovado',
            'Aprovado','Recuperação','Aprovado','Reprovado','Aprovado',
            'Recuperação','Aprovado','Reprovado','Aprovado','Reprovado',
            'Recuperação','Aprovado','Recuperação','Aprovado','Reprovado',
            'Aprovado','Recuperação','Aprovado','Reprovado','Recuperação',
            'Reprovado','Aprovado','Recuperação','Aprovado','Reprovado',
            'Aprovado','Recuperação','Aprovado','Recuperação','Reprovado'
        ]
    }
    return pd.DataFrame(dados)


# ------------------------------------------------------------
# 2. Treinamento do modelo (cacheado — treina só uma vez)
# ------------------------------------------------------------
@st.cache_resource
def treinar_modelo(df):
    x = df[['Horas_de_estudo', 'Faltas', 'Nota']]
    y = df['Situacao']

    x_train, x_teste, y_train, y_teste = train_test_split(
        x, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    modelo = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42
    )
    modelo.fit(x_train, y_train)

    y_pred = modelo.predict(x_teste)
    acuracia = accuracy_score(y_teste, y_pred)
    relatorio = classification_report(y_teste, y_pred, zero_division=0)

    return modelo, acuracia, relatorio


df = carregar_dados()
modelo, acuracia, relatorio = treinar_modelo(df)

# ------------------------------------------------------------
# 3. Interface Streamlit
# ------------------------------------------------------------
st.title("🎓 Preditor de Situação do Aluno")
st.markdown(
    """
    Informe as horas semanais de estudo, o número de faltas e a nota
    do aluno para estimar a probabilidade de **Aprovado**,
    **Recuperação** ou **Reprovado**.
    """
)

col1, col2 = st.columns(2)

with col1:
    horas = st.slider("Horas de estudo por semana", min_value=0, max_value=20, value=6, step=1)
    faltas = st.slider("Número de faltas", min_value=0, max_value=30, value=2, step=1)
    nota = st.slider("Nota", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
    prever = st.button("Prever situação", type="primary", use_container_width=True)

with col2:
    st.markdown("**Exemplos rápidos**")
    exemplos = {
        "Bom desempenho": (10, 1, 9.0),
        "Desempenho médio": (6, 2, 7.0),
        "Risco de recuperação": (4, 10, 5.5),
        "Risco de reprovação": (2, 18, 3.0),
    }
    exemplo_escolhido = st.selectbox("Escolha um exemplo", ["Nenhum"] + list(exemplos.keys()))
    if exemplo_escolhido != "Nenhum":
        horas, faltas, nota = exemplos[exemplo_escolhido]
        st.info(f"Horas: {horas} | Faltas: {faltas} | Nota: {nota}")
        prever = True

if prever:
    df_novo = pd.DataFrame([[horas, faltas, nota]], columns=['Horas_de_estudo', 'Faltas', 'Nota'])
    probabilidades = modelo.predict_proba(df_novo)[0]
    classes = modelo.classes_

    previsao = modelo.predict(df_novo)[0]
    st.subheader(f"Resultado previsto: **{previsao}**")

    df_prob = pd.DataFrame({
        "Situação": classes,
        "Probabilidade": probabilidades
    }).sort_values("Probabilidade", ascending=False).reset_index(drop=True)

    st.bar_chart(df_prob.set_index("Situação"))
    st.dataframe(
        df_prob.style.format({"Probabilidade": "{:.1%}"}),
        use_container_width=True,
        hide_index=True
    )

# ------------------------------------------------------------
# 4. Informações sobre o modelo (expansível)
# ------------------------------------------------------------
with st.expander("📊 Ver base de dados e avaliação do modelo"):
    st.write(f"Total de alunos na base: {len(df)}")
    st.write(df['Situacao'].value_counts())
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown(f"**Acurácia no conjunto de teste:** {acuracia:.1%}")
    st.text("Relatório de classificação:")
    st.text(relatorio)
