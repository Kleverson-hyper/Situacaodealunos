# ============================================================
# Preditor de Situação do Aluno — versão melhorada
# ============================================================
# Melhorias em relação à versão original:
#  1. Base de dados muito maior (60 alunos) e mais variada,
#     cobrindo bem as 3 classes (Aprovado / Recuperação / Reprovado).
#  2. Divisão treino/teste estratificada (mantém a proporção
#     das classes) e avaliação do modelo (acurácia + relatório).
#  3. Modelo trocado para RandomForestClassifier (mais robusto
#     e menos propenso a "decorar" os dados que uma única árvore).
#  4. Interface Gradio bem mais completa: sliders com faixas
#     realistas, título, descrição, exemplos prontos e saída
#     mostrando a probabilidade de cada situação (não só a classe).
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import gradio as gr

# ------------------------------------------------------------
# 1. Base de dados (muito mais ampla e mais realista)
# ------------------------------------------------------------
# Regra usada para gerar os dados (com um pouco de ruído):
#   - Poucas faltas + muitas horas de estudo + nota alta  -> Aprovado
#   - Situação intermediária                              -> Recuperação
#   - Muitas faltas e/ou nota baixa                       -> Reprovado

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

df = pd.DataFrame(dados)
print("Total de alunos na base:", len(df))
print(df['Situacao'].value_counts())
print(df.head())

# ------------------------------------------------------------
# 2. Separação treino/teste (estratificada)
# ------------------------------------------------------------
x = df[['Horas_de_estudo', 'Faltas', 'Nota']]
y = df['Situacao']  # Series (não DataFrame) — evita warnings do sklearn

x_train, x_teste, y_train, y_teste = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # mantém a proporção de cada situação no treino e no teste
)

# ------------------------------------------------------------
# 3. Modelo: Random Forest (mais robusto que uma única árvore)
# ------------------------------------------------------------
modelo = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42
)
modelo.fit(x_train, y_train)

# ------------------------------------------------------------
# 4. Avaliação do modelo
# ------------------------------------------------------------
y_pred = modelo.predict(x_teste)
print("\nAcurácia no conjunto de teste:", accuracy_score(y_teste, y_pred))
print("\nRelatório de classificação:\n", classification_report(y_teste, y_pred, zero_division=0))

# Teste rápido com um aluno fictício
novo_aluno = pd.DataFrame([[6, 2, 7.0]], columns=['Horas_de_estudo', 'Faltas', 'Nota'])
previsao = modelo.predict(novo_aluno)
print(f"\nO sistema previu: {previsao[0]}")

# ------------------------------------------------------------
# 5. Interface Gradio melhorada
# ------------------------------------------------------------
def prever_situacao(horas, faltas, nota):
    df_novo = pd.DataFrame(
        [[horas, faltas, nota]],
        columns=['Horas_de_estudo', 'Faltas', 'Nota']
    )
    probabilidades = modelo.predict_proba(df_novo)[0]
    classes = modelo.classes_
    # Retorna um dicionário {classe: probabilidade}, que o gr.Label
    # exibe como um gráfico de barras com as porcentagens
    return {classe: float(prob) for classe, prob in zip(classes, probabilidades)}

with gr.Blocks(title="Preditor de Situação do Aluno") as interface:
    gr.Markdown(
        """
        # 🎓 Preditor de Situação do Aluno
        Informe as horas semanais de estudo, o número de faltas e a nota
        do aluno para estimar a probabilidade de **Aprovado**,
        **Recuperação** ou **Reprovado**.
        """
    )

    with gr.Row():
        with gr.Column():
            horas_input = gr.Slider(0, 20, value=6, step=1, label="Horas de estudo por semana")
            faltas_input = gr.Slider(0, 30, value=2, step=1, label="Número de faltas")
            nota_input = gr.Slider(0, 10, value=7.0, step=0.1, label="Nota")
            botao = gr.Button("Prever situação", variant="primary")
        with gr.Column():
            saida = gr.Label(num_top_classes=3, label="Probabilidade por situação")

    botao.click(fn=prever_situacao, inputs=[horas_input, faltas_input, nota_input], outputs=saida)

    gr.Examples(
        examples=[
            [10, 1, 9.0],
            [6, 2, 7.0],
            [4, 10, 5.5],
            [2, 18, 3.0],
        ],
        inputs=[horas_input, faltas_input, nota_input],
        outputs=saida,
        fn=prever_situacao,
        cache_examples=False,
        label="Exemplos"
    )

if __name__ == "__main__":
    interface.launch()
