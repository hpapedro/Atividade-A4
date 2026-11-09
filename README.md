# Atividade de Machine Learning — Classificação e Regressão

**Aluno:** Pedro  
**Arquivo principal:** `Atividade_ML_Pedro.py`

## Sobre o Projeto

Este projeto reúne dois modelos de Machine Learning em um único script Python (`.py`):

1. **Parte 1 — Classificação (Câncer de Mama):**
   - Usa o dataset *Breast Cancer Wisconsin* (`569` registros e `30` atributos) do Scikit-Learn.
   - Padroniza os dados (`StandardScaler`) e treina uma **Regressão Logística** para classificar tumores como **Malignos (`0`)** ou **Benignos (`1`)**.
   - Avalia o modelo com **Accuracy (98,25%), Precision, Recall, F1-Score** e gera a **Matriz de Confusão** (`parte1_matriz_confusao.png`).

2. **Parte 2 — Regressão Linear (Preço de Imóveis):**
   - Usa o dataset *California Housing* (`20.640` registros) do Scikit-Learn.
   - Faz a análise exploratória gerando o histograma de preços (`parte2_histograma_preco.png`) e o mapa de correlação (`parte2_mapa_correlacao.png`).
   - Treina uma **Regressão Linear Múltipla** para prever o valor mediano dos imóveis (`MedHouseVal`, em unidades de US$ 100 mil).
   - Avalia o modelo com **MAE, MSE, RMSE e R² (`0.5758`)** e gera o gráfico **Real vs. Previsto** (`parte2_real_vs_previsto.png`).

---

## Como Funciona o Código (`Atividade_ML_Pedro.py`)

Quando executado, o script roda automaticamente as **8 etapas obrigatórias** em sequência:

1. **Introdução:** Apresenta o objetivo dos dois problemas no terminal.
2. **Importação:** Carrega `pandas`, `numpy`, `matplotlib`, `seaborn` e `scikit-learn`.
3. **Carregamento dos Dados:** Baixa/carrega os datasets das Partes 1 e 2.
4. **Análise Exploratória (EDA):** Mostra dimensões (`shape`), estatísticas (`describe`), valores nulos e salva os gráficos exploratórios.
5. **Pré-processamento:** Separa `X` (atributos) e `y` (alvo), divide em treino/teste (`80%/20%`) e padroniza os dados da Parte 1.
6. **Treinamento:** Ajusta os modelos `LogisticRegression` (Parte 1) e `LinearRegression` (Parte 2).
7. **Avaliação e Gráficos:** Calcula todas as métricas, salva os 4 gráficos em `.png` na pasta e exibe a interpretação dos resultados no terminal.
8. **Conclusão:** Exibe a análise crítica final sobre as limitações e melhorias de cada modelo.

---

## Como Executar

1. Instale as bibliotecas necessárias:
   ```powershell
   pip install -r requirements.txt
   ```

2. Execute o script:
   ```powershell
   python Atividade_ML_Pedro.py
   ```
   *(Ou `py -3.12 Atividade_ML_Pedro.py`)*

Ao finalizar, todos os resultados e interpretações aparecem no terminal e os 4 gráficos (`.png`) ficam salvos na mesma pasta.
