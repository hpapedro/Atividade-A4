#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ATIVIDADE DE MACHINE LEARNING: CLASSIFICAÇÃO + REGRESSÃO COM PYTHON
Aluno: Pedro
Arquivo: Atividade_ML_Pedro.py
================================================================================

1. INTRODUÇÃO
-------------
Este script desenvolve dois mini-projetos práticos de Machine Learning
supervisionado utilizando a biblioteca Scikit-Learn:

- Parte 1 — Classificação (Diagnóstico de Câncer de Mama):
  Utiliza o dataset Breast Cancer Wisconsin (569 registros e 30 atributos
  morfológicos de núcleos celulares) para classificar tumores como Malignos (0)
  ou Benignos (1) por meio de Regressão Logística com dados padronizados.
  O foco analítico recai sobre a sensibilidade (Recall) na detecção de casos
  malignos e a minimização de Falsos Negativos.

- Parte 2 — Regressão Linear (Previsão de Preço de Imóveis):
  Utiliza o dataset California Housing (20.640 registros e 8 variáveis
  socioeconômicas/geográficas) para estimar o valor mediano das residências
  (MedHouseVal, expresso em unidades de US$ 100 mil) usando Regressão Linear
  Múltipla. Avalia-se a qualidade do ajuste pelas métricas MAE, MSE, RMSE e R²,
  além da análise crítica sobre coeficientes, outliers e o teto de preços.
================================================================================
"""

# %% [2. Importação das bibliotecas]
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Configuração visual padrão para os gráficos
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (9, 6)
plt.rcParams["figure.dpi"] = 120

OUTPUT_DIR = Path(__file__).resolve().parent


def exibir_titulo(texto: str) -> None:
    """Imprime um cabeçalho formatado no terminal para organização da saída."""
    separador = "=" * 80
    print(f"\n{separador}\n{texto}\n{separador}")


# ==============================================================================
# PARTE 1 - CLASSIFICAÇÃO (0,5 ponto)
# Problema: Classificar tumores como benigno ou maligno (Breast Cancer Wisconsin)
# ==============================================================================


def executar_parte_1_classificacao() -> dict:
    exibir_titulo("1. INTRODUÇÃO DO TRABALHO")
    print(
        "Objetivo geral:\n"
        "  Aplicar o fluxo completo de Ciência de Dados (carregamento, análise\n"
        "  exploratória, pré-processamento, treinamento, avaliação por métricas\n"
        "  e interpretação crítica) em dois problemas clássicos:\n"
        "  1) Classificação binária médica (Breast Cancer Wisconsin).\n"
        "  2) Regressão linear imobiliária (California Housing)."
    )

    # %% [3.1 Carregamento dos dados - Parte 1]
    exibir_titulo("PARTE 1: CLASSIFICAÇÃO — 3.1 CARREGAMENTO DOS DADOS")
    cancer_data = load_breast_cancer()

    # Criação do DataFrame com os 30 atributos originais
    df_cancer = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
    df_cancer["target"] = cancer_data.target

    # Atenção do Guia: no sklearn, 0 = malignant (maligno) e 1 = benign (benigno)
    rotulos_pt = {0: "Maligno (0 - malignant)", 1: "Benigno (1 - benign)"}
    df_cancer["classe_nome"] = df_cancer["target"].map(rotulos_pt)

    print("Dataset Breast Cancer Wisconsin carregado com sucesso!")
    print(f"Nomes originais das classes (target_names): {list(cancer_data.target_names)}")
    print("Mapeamento: 0 -> Maligno (malignant) | 1 -> Benigno (benign)")

    # %% [4.1 Análise Exploratória (EDA) - Parte 1]
    exibir_titulo("PARTE 1: CLASSIFICAÇÃO — 4.1 ANÁLISE EXPLORATÓRIA (EDA)")

    num_registros, num_atributos = cancer_data.data.shape
    print(f"Formato da matriz de atributos (X.shape): {cancer_data.data.shape}")
    print(f"  -> Nº de registros: {num_registros} (esperado pelo guia: 569)")
    print(f"  -> Nº de atributos: {num_atributos} (esperado pelo guia: 30)")

    print("\nDistribuição das classes (value_counts):")
    contagem_classes = df_cancer["classe_nome"].value_counts()
    for classe, qtd in contagem_classes.items():
        percentual = (qtd / num_registros) * 100
        print(f"  - {classe}: {qtd} registros ({percentual:.2f}%)")

    print("\nEstatísticas descritivas dos primeiros 6 atributos (describe()):")
    print(df_cancer[cancer_data.feature_names[:6]].describe().T[["mean", "std", "min", "50%", "max"]].round(3))

    # %% [5.1 Pré-processamento - Parte 1]
    exibir_titulo("PARTE 1: CLASSIFICAÇÃO — 5.1 PRÉ-PROCESSAMENTO")

    # Separação entre variáveis explicativas (X) e variável-alvo (y)
    X = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
    y = pd.Series(cancer_data.target, name="target")

    # Divisão treino/teste (80% treino / 20% teste) com estratificação e semente fixa
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Tamanho do conjunto de Treino (80%): {X_train.shape[0]} amostras")
    print(f"Tamanho do conjunto de Teste  (20%): {X_test.shape[0]} amostras")
    print(
        f"Proporção no teste: {(y_test == 0).sum()} malignos (0) e "
        f"{(y_test == 1).sum()} benignos (1)."
    )

    # Padronização (StandardScaler): obrigatória para Regressão Logística / KNN,
    # pois os atributos possuem escalas muito diferentes (ex.: 'mean area' ~654 vs 'mean smoothness' ~0.096).
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(
        "\nPadronização (StandardScaler) aplicada com sucesso!\n"
        "  -> Ajuste (fit) feito apenas no treino para evitar vazamento de dados (data leakage)."
    )

    # %% [6.1 Treinamento do Modelo - Parte 1]
    exibir_titulo("PARTE 1: CLASSIFICAÇÃO — 6.1 TREINAMENTO DO MODELO")

    modelo_clf = LogisticRegression(random_state=42, max_iter=1000)
    modelo_clf.fit(X_train_scaled, y_train)
    print("Modelo escolhido e treinado: Regressão Logística (LogisticRegression).")

    # %% [7.1 Avaliação dos Resultados e Análise Breve - Parte 1]
    exibir_titulo("PARTE 1: CLASSIFICAÇÃO — 7.1 AVALIAÇÃO E INTERPRETAÇÃO")

    y_pred = modelo_clf.predict(X_test_scaled)

    # Cálculo das 5 métricas solicitadas
    acc = accuracy_score(y_test, y_pred)
    # Métricas focadas na classe clínica mais crítica (0 = maligno) e médias ponderadas
    prec_maligno = precision_score(y_test, y_pred, pos_label=0)
    rec_maligno = recall_score(y_test, y_pred, pos_label=0)
    f1_maligno = f1_score(y_test, y_pred, pos_label=0)

    prec_macro = precision_score(y_test, y_pred, average="macro")
    rec_macro = recall_score(y_test, y_pred, average="macro")
    f1_macro = f1_score(y_test, y_pred, average="macro")

    nomes_classes = ["Maligno (0 - malignant)", "Benigno (1 - benign)"]
    matriz_conf = confusion_matrix(y_test, y_pred, labels=[0, 1])

    # Decomposição da matriz considerando 0 (Maligno) como primeira linha/coluna:
    # Linha 0 (Real = Maligno): [Verdadeiro Maligno, Falso Benigno (Falso Negativo clínico)]
    # Linha 1 (Real = Benigno): [Falso Maligno (Falso Positivo clínico), Verdadeiro Benigno]
    verdadeiros_malignos = matriz_conf[0, 0]
    falsos_negativos_maligno = matriz_conf[0, 1]  # Tumor maligno previsto como benigno!
    falsos_positivos_maligno = matriz_conf[1, 0]  # Tumor benigno previsto como maligno
    verdadeiros_benignos = matriz_conf[1, 1]

    print("Resumo das 5 Métricas Principais (Conjunto de Teste - 114 pacientes):")
    print(f"  1. Accuracy (Acurácia Global)         : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"  2. Precision (Classe Maligna - 0)     : {prec_maligno:.4f} | Macro: {prec_macro:.4f}")
    print(f"  3. Recall    (Classe Maligna - 0)     : {rec_maligno:.4f} | Macro: {rec_macro:.4f}")
    print(f"  4. F1-Score  (Classe Maligna - 0)     : {f1_maligno:.4f} | Macro: {f1_macro:.4f}")
    print("\n  5. Matriz de Confusão (Linhas = Real | Colunas = Previsto):")
    df_cm = pd.DataFrame(
        matriz_conf,
        index=["Real: Maligno (0)", "Real: Benigno (1)"],
        columns=["Prev: Maligno (0)", "Prev: Benigno (1)"],
    )
    print(df_cm)

    print("\nRelatório Completo de Classificação (classification_report):")
    print(classification_report(y_test, y_pred, target_names=nomes_classes, digits=4))

    # Geração do gráfico da Matriz de Confusão
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        matriz_conf,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Maligno (0)", "Benigno (1)"],
        yticklabels=["Maligno (0)", "Benigno (1)"],
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"},
        ax=ax,
    )
    ax.set_title("Parte 1 — Matriz de Confusão (Regressão Logística)", fontsize=13, pad=12)
    ax.set_xlabel("Classe Prevista pelo Modelo", fontsize=11)
    ax.set_ylabel("Classe Real (Diagnóstico Real)", fontsize=11)
    plt.tight_layout()
    caminho_cm = OUTPUT_DIR / "parte1_matriz_confusao.png"
    plt.savefig(caminho_cm)
    plt.close(fig)
    print(f"[Gráfico salvo em: {caminho_cm.name}]")

    # Interpretação escrita e análise breve exigida pelo guia
    print(
        "\n--- ANÁLISE BREVE E INTERPRETAÇÃO DOS RESULTADOS (PARTE 1) ---\n"
        f"1. Desempenho Geral: O modelo de Regressão Logística padronizado obteve excelente\n"
        f"   desempenho, alcançando {acc * 100:.2f}% de acurácia no conjunto de teste (acertou\n"
        f"   {verdadeiros_malignos + verdadeiros_benignos} de {len(y_test)} diagnósticos).\n"
        f"2. Análise de Falsos Negativos (Crítico): Na matriz de confusão, observamos apenas\n"
        f"   {falsos_negativos_maligno} caso(s) de tumor REALMENTE Maligno (0) classificado erroneamente como\n"
        f"   Benigno (1), contra {verdadeiros_malignos} tumores malignos detectados corretamente.\n"
        f"   Já entre os {verdadeiros_benignos + falsos_positivos_maligno} casos benignos reais, "
        f"{verdadeiros_benignos} foram previstos corretamente e houve {falsos_positivos_maligno} falso(s) alarme(s).\n"
        f"3. Importância do Recall na Classe Maligna: Em diagnóstico médico oncológico, o\n"
        f"   Recall da classe maligna ({rec_maligno * 100:.2f}%) é a métrica mais crítica porque um Falso\n"
        f"   Negativo (dizer que o tumor é benigno quando na verdade é maligno) pode atrasar\n"
        f"   o início do tratamento e colocar a vida do paciente em risco. Se desejarmos\n"
        f"   reduzir os Falsos Negativos a zero, podemos ajustar o limiar de probabilidade\n"
        f"   (threshold) do classificador para priorizar ainda mais a sensibilidade da classe 0."
    )

    return {
        "accuracy": acc,
        "recall_maligno": rec_maligno,
        "falsos_negativos": int(falsos_negativos_maligno),
    }


# ==============================================================================
# PARTE 2 - REGRESSÃO LINEAR (0,5 ponto)
# Problema: Prever o preço de imóveis (California Housing Dataset)
# ==============================================================================


def executar_parte_2_regressao() -> dict:
    # %% [3.2 Carregamento dos dados - Parte 2]
    exibir_titulo("PARTE 2: REGRESSÃO LINEAR — 3.2 CARREGAMENTO DOS DADOS")

    california = fetch_california_housing(as_frame=True)
    df_housing = california.frame.copy()

    print("Dataset California Housing carregado com sucesso!")
    print(f"Dimensões do DataFrame: {df_housing.shape[0]} registros e {df_housing.shape[1]} colunas.")
    print("Unidade da variável-alvo (MedHouseVal): centenas de milhares de dólares (US$ 100.000).")

    # %% [4.2 Análise Exploratória (EDA) - Parte 2]
    exibir_titulo("PARTE 2: REGRESSÃO LINEAR — 4.2 ANÁLISE EXPLORATÓRIA (EDA)")

    print("1) Informações estruturais do DataFrame (info()):")
    df_housing.info()

    print("\n2) Verificação de valores nulos por coluna (isnull().sum()):")
    nulos = df_housing.isnull().sum()
    print(nulos)
    print(f"Total de valores ausentes no dataset: {int(nulos.sum())}")

    print("\n3) Estatísticas descritivas (describe()):")
    print(df_housing.describe().T[["count", "mean", "std", "min", "50%", "max"]].round(3))

    # Observação importante sobre o teto (capping) de MedHouseVal = 5.00001
    qtd_teto = (df_housing["MedHouseVal"] >= 5.0).sum()
    pct_teto = (qtd_teto / len(df_housing)) * 100
    print(
        f"\nObservação de EDA: Existem {qtd_teto} imóveis ({pct_teto:.2f}% do dataset) cravados no\n"
        "valor máximo de MedHouseVal = 5.0 (US$ 500.000), evidenciando censura/teto nos dados."
    )

    # Gráfico 1 da EDA: Histograma do Preço (MedHouseVal)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    sns.histplot(df_housing["MedHouseVal"], bins=50, kde=True, color="#2b5c8f", ax=ax)
    ax.axvline(5.0, color="crimson", linestyle="--", linewidth=2, label="Teto em 5.0 (US$ 500 mil)")
    ax.set_title("Parte 2 (EDA) — Distribuição do Valor Mediano dos Imóveis (MedHouseVal)", fontsize=12)
    ax.set_xlabel("Valor Mediano do Imóvel (em US$ 100 mil)", fontsize=11)
    ax.set_ylabel("Frequência (Nº de Quarteirões/Blocos)", fontsize=11)
    ax.legend()
    plt.tight_layout()
    caminho_hist = OUTPUT_DIR / "parte2_histograma_preco.png"
    plt.savefig(caminho_hist)
    plt.close(fig)
    print(f"[Gráfico de Histograma salvo em: {caminho_hist.name}]")

    # Gráfico 2 da EDA: Mapa de Correlação (Seaborn Heatmap)
    matriz_corr = df_housing.corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        matriz_corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Parte 2 (EDA) — Mapa de Correlação das Variáveis (California Housing)", fontsize=12)
    plt.tight_layout()
    caminho_corr = OUTPUT_DIR / "parte2_mapa_correlacao.png"
    plt.savefig(caminho_corr)
    plt.close(fig)
    print(f"[Mapa de Correlação salvo em: {caminho_corr.name}]")

    # %% [5.2 Pré-processamento e Definição de Variáveis - Parte 2]
    exibir_titulo("PARTE 2: REGRESSÃO LINEAR — 5.2 DEFINIÇÃO DE VARIÁVEIS E SPLIT")

    # Variável dependente (alvo): MedHouseVal | Independentes: demais 8 colunas
    X = df_housing.drop(columns=["MedHouseVal"])
    y = df_housing["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print(f"Variável dependente (y): 'MedHouseVal' (em unidades de US$ 100 mil)")
    print(f"Variáveis independentes (X): {list(X.columns)}")
    print(f"Divisão Treino (80%): {X_train.shape[0]} registros | Teste (20%): {X_test.shape[0]} registros")

    # %% [6.2 Treinamento do Modelo de Regressão Linear - Parte 2]
    exibir_titulo("PARTE 2: REGRESSÃO LINEAR — 6.2 TREINAMENTO DO MODELO")

    modelo_reg = LinearRegression()
    modelo_reg.fit(X_train, y_train)

    print("Modelo LinearRegression ajustado com sucesso!")
    print(f"Intercepto (beta_0): {modelo_reg.intercept_:.4f}")

    df_coefs = (
        pd.DataFrame({"Variável": X.columns, "Coeficiente": modelo_reg.coef_})
        .assign(Impacto_Absoluto=lambda d: d["Coeficiente"].abs())
        .sort_values(by="Impacto_Absoluto", ascending=False)
    )
    print("\nCoeficientes da Regressão Linear (ordenados por magnitude):")
    print(df_coefs[["Variável", "Coeficiente"]].to_string(index=False))

    # %% [7.2 Avaliação dos Resultados, Gráfico Real x Previsto e Interpretação - Parte 2]
    exibir_titulo("PARTE 2: REGRESSÃO LINEAR — 7.2 AVALIAÇÃO, GRÁFICO E INTERPRETAÇÃO")

    y_pred = modelo_reg.predict(X_test)

    # Cálculo das 4 métricas obrigatórias de regressão
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    mae_usd = mae * 100_000
    rmse_usd = rmse * 100_000

    print("Métricas de Desempenho no Conjunto de Teste:")
    print(f"  - MAE  (Erro Absoluto Médio)       : {mae:.4f}  (~US$ {mae_usd:,.2f})")
    print(f"  - MSE  (Erro Quadrático Médio)     : {mse:.4f}")
    print(f"  - RMSE (Raiz do Erro Quadrático)   : {rmse:.4f}  (~US$ {rmse_usd:,.2f})")
    print(f"  - R²   (Coeficiente de Determinação): {r2:.4f}  ({r2 * 100:.2f}% da variância explicada)")

    # Gráfico obrigatório: Dispersão Real x Previsto com linha de referência y = x
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.scatter(
        y_test,
        y_pred,
        alpha=0.25,
        color="#1f77b4",
        edgecolors="none",
        s=22,
        label="Imóveis no conjunto de teste",
    )

    lim_min = 0.0
    lim_max = 5.5
    ax.plot(
        [lim_min, lim_max],
        [lim_min, lim_max],
        color="crimson",
        linestyle="--",
        linewidth=2.2,
        label="Linha de Referência Ideal (y = x)",
    )
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(-0.5, 6.5)
    ax.set_title(
        f"Parte 2 — Preço Real vs. Preço Previsto (LinearRegression | R² = {r2:.3f})",
        fontsize=12,
        pad=12,
    )
    ax.set_xlabel("Preço Real — MedHouseVal (unidades de US$ 100 mil)", fontsize=11)
    ax.set_ylabel("Preço Previsto — y_pred (unidades de US$ 100 mil)", fontsize=11)
    ax.legend(loc="upper left")
    plt.tight_layout()
    caminho_scatter = OUTPUT_DIR / "parte2_real_vs_previsto.png"
    plt.savefig(caminho_scatter)
    plt.close(fig)
    print(f"\n[Gráfico Real vs. Previsto salvo em: {caminho_scatter.name}]")

    # Interpretação detalhada exigida pelo Guia
    print(
        "\n--- INTERPRETAÇÃO DAS MÉTRICAS E ANÁLISE DOS RESULTADOS (PARTE 2) ---\n"
        f"1. Significado do MAE e Escala em Dólares:\n"
        f"   O MAE de {mae:.4f} indica que, em média, as previsões do modelo erram em {mae:.4f}\n"
        f"   unidades de US$ 100 mil — o que equivale a aproximadamente US$ {mae_usd:,.0f} de erro\n"
        f"   absoluto médio por região censitária.\n"
        f"2. Significado do MSE e RMSE:\n"
        f"   O MSE ({mse:.4f}) eleva os erros ao quadrado, penalizando fortemente grandes desvios\n"
        f"   (outliers). Extraindo a raiz quadrada, temos o RMSE de {rmse:.4f} (~US$ {rmse_usd:,.0f}),\n"
        f"   que volta para a mesma escala do preço e mostra que erros extremos puxam a\n"
        f"   dispersão para cima em relação ao MAE.\n"
        f"3. Poder Explicativo (R² = {r2:.4f}):\n"
        f"   O modelo linear múltiplo explica cerca de {r2 * 100:.1f}% da variação dos preços dos\n"
        f"   imóveis. Conforme destacado no guia, um R² em torno de 0,60 reflete o limite natural\n"
        f"   de um modelo puramente linear diante de dinâmicas imobiliárias e espaciais não-lineares.\n"
        f"4. Variáveis de Maior Influência (Coeficientes):\n"
        f"   - 'MedInc' (Renda Mediana, coef = +0.4487) possui a maior correlação positiva (0.69)\n"
        f"     com o preço: cada aumento de 1 unidade na renda eleva o preço previsto em ~US$ 44,9 mil.\n"
        f"   - 'AveBedrms' (+0.7831) e 'AveRooms' (-0.1233) atuam em conjunto ajustando a proporção\n"
        f"     de dormitórios por cômodo.\n"
        f"   - 'Latitude' (-0.4198) e 'Longitude' (-0.4337) capturam o gradiente geográfico\n"
        f"     (imóveis mais próximos ao litoral sul/oeste da Califórnia são mais caros).\n"
        f"5. Teto de Preços e Outliers:\n"
        f"   No gráfico Real vs. Previsto, nota-se uma faixa vertical densa em Real = 5.0\n"
        f"   (US$ 500 mil). Como o censo truncou imóveis mais caros nesse teto, a reta linear\n"
        f"   tende a subestimar imóveis de altíssimo padrão. Além disso, valores extremos em\n"
        f"   'AveRooms' (máximo de ~141 cômodos) e 'AveOccup' (~1.243) funcionam como outliers\n"
        f"   que distorcem a inclinação da reta de mínimos quadrados."
    )

    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


# ==============================================================================
# 8. CONCLUSÃO — ANÁLISE CRÍTICA FINAL DOS DOIS PROBLEMAS
# ==============================================================================


def exibir_conclusao_final(res_parte1: dict, res_parte2: dict) -> None:
    exibir_titulo("8. CONCLUSÃO — ANÁLISE CRÍTICA FINAL DOS DOIS PROBLEMAS")
    print(
        "Síntese Comparativa e Limitações:\n"
        "---------------------------------\n"
        f"1. Parte 1 (Classificação — Breast Cancer Wisconsin):\n"
        f"   - A Regressão Logística com atributos padronizados atingiu {res_parte1['accuracy'] * 100:.2f}% de\n"
        f"     acurácia e {res_parte1['recall_maligno'] * 100:.2f}% de Recall para tumores malignos (apenas\n"
        f"     {res_parte1['falsos_negativos']} falso negativo em 114 exames de teste).\n"
        f"   - Limitações: O dataset é relativamente pequeno (569 amostras) e apresenta forte\n"
        f"     multicolinearidade entre variáveis geométricas (raio, perímetro e área).\n"
        f"   - Possíveis melhorias: Ajustar o threshold de decisão (ex.: de 0.50 para 0.35)\n"
        f"     para zerar os Falsos Negativos clínicos, aplicar regularização L1/L2 (GridSearchCV)\n"
        f"     e validar com Stratified K-Fold Cross-Validation.\n\n"
        f"2. Parte 2 (Regressão Linear — California Housing):\n"
        f"   - A Regressão Linear Múltipla obteve R² = {res_parte2['r2']:.4f} e MAE = {res_parte2['mae']:.4f}\n"
        f"     (~US$ {res_parte2['mae'] * 100_000:,.0f}), capturando a tendência principal liderada pela\n"
        f"     renda mediana (MedInc), mas deixando ~{(1 - res_parte2['r2']) * 100:.1f}% da variância sem explicação.\n"
        f"   - Limitações: (a) Incapacidade de modelar fronteiras geográficas não-lineares\n"
        f"     apenas com combinações lineares de Latitude e Longitude; (b) sensibilidade a\n"
        f"     outliers extremos em AveRooms/AveOccup; e (c) censura artificial de preços em\n"
        f"     MedHouseVal = 5.0 (US$ 500 mil).\n"
        f"   - Possíveis melhorias: Utilizar modelos baseados em árvores (Random Forest Regressor,\n"
        f"     HistGradientBoosting ou XGBoost, que superam R² > 0,80 nesse dataset), remover ou\n"
        f"     aplicar transformação logarítmica em variáveis assimétricas e criar variáveis de\n"
        f"     interação espacial."
    )
    exibir_titulo("EXECUÇÃO CONCLUÍDA COM SUCESSO!")


if __name__ == "__main__":
    resultados_p1 = executar_parte_1_classificacao()
    resultados_p2 = executar_parte_2_regressao()
    exibir_conclusao_final(resultados_p1, resultados_p2)

