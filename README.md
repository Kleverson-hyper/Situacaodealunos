# 🎓 Preditor de Situação do Aluno

Aplicação de Machine Learning que prevê se um aluno será **Aprovado**,
ficará em **Recuperação** ou será **Reprovado**, com base em três
variáveis: horas de estudo por semana, número de faltas e nota.

O modelo é treinado com um `RandomForestClassifier` (scikit-learn) e
exposto através de uma interface web interativa criada com
[Gradio](https://www.gradio.app/).

## ✨ Funcionalidades

- Base de dados sintética com **60 alunos**, balanceada entre as três
  situações possíveis.
- Divisão treino/teste **estratificada**, preservando a proporção das
  classes.
- Modelo `RandomForestClassifier` (200 árvores, profundidade limitada),
  mais robusto que uma única árvore de decisão.
- Avaliação automática do modelo (acurácia e relatório de
  classificação) exibida no console ao rodar o script.
- Interface Gradio com:
  - Sliders com faixas realistas (horas de estudo, faltas, nota);
  - Exibição da **probabilidade de cada situação**, não apenas da
    classe prevista;
  - Exemplos prontos para teste rápido.

## 📋 Pré-requisitos

- Python 3.9 ou superior
- Bibliotecas:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `gradio`

## 🔧 Instalação

```bash
pip install pandas numpy scikit-learn gradio
```

## ▶️ Como executar

```bash
python preditor_situacao_aluno.py
```

Ao rodar, o script:

1. Monta a base de dados e mostra um resumo no console (quantidade de
   alunos por situação e as primeiras linhas da tabela).
2. Treina o modelo e imprime a acurácia e o relatório de classificação
   obtidos no conjunto de teste.
3. Faz uma previsão de exemplo para um aluno fictício.
4. Abre a interface Gradio no navegador (geralmente em
   `http://127.0.0.1:7860`).

## 🖥️ Usando a interface

Na interface, ajuste os três sliders:

| Campo | Faixa |
|---|---|
| Horas de estudo por semana | 0 a 20 |
| Número de faltas | 0 a 30 |
| Nota | 0 a 10 |

Clique em **"Prever situação"** para ver a probabilidade estimada de o
aluno ficar em cada uma das três situações (Aprovado, Recuperação,
Reprovado). Também é possível clicar em um dos **exemplos** prontos
para testar rapidamente.

## 🧠 Sobre os dados e o modelo

Os dados usados neste projeto são **sintéticos**, gerados seguindo uma
lógica simples:

- Poucas faltas + muitas horas de estudo + nota alta → tende a
  **Aprovado**;
- Muitas faltas e/ou nota baixa → tende a **Reprovado**;
- Situações intermediárias → tende a **Recuperação**.

Por serem sintéticos, os resultados servem para fins didáticos e de
demonstração da lógica de classificação. Para uso em um cenário real,
recomenda-se substituir a base de dados por registros reais de alunos
(por exemplo, importando de uma planilha ou de um sistema acadêmico).

## 📁 Estrutura do arquivo

```
preditor_situacao_aluno.py
├── 1. Base de dados (DataFrame com 60 alunos)
├── 2. Separação treino/teste (estratificada)
├── 3. Treinamento do modelo (RandomForestClassifier)
├── 4. Avaliação do modelo (acurácia + classification_report)
└── 5. Interface Gradio (gr.Blocks com sliders e exemplos)
```

## 📌 Possíveis melhorias futuras

- Substituir os dados sintéticos por dados reais de alunos.
- Adicionar mais variáveis (ex: participação em aula, entrega de
  trabalhos, disciplina cursada).
- Persistir o modelo treinado em disco (`joblib`) para não precisar
  retreinar a cada execução.
- Publicar a interface Gradio online (ex: Hugging Face Spaces).
