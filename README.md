# Predição do Preço de Venda de Veículos Ford com Machine Learning

## Sobre o projeto

Este projeto tem como objetivo desenvolver um modelo de regressão capaz de prever o preço de venda de veículos da marca Ford a partir de suas características técnicas.

Além da construção do modelo preditivo, o principal foco do estudo é comparar duas abordagens de pré-processamento para variáveis categóricas e analisar qual delas produz melhor desempenho em um problema de regressão.

As estratégias avaliadas são:

- **One-Hot Encoding (Dummies)** utilizando `pd.get_dummies()`;
- **Label Encoding** utilizando `LabelEncoder` do Scikit-Learn.

Após o pré-processamento, diferentes modelos de regressão serão treinados e comparados por meio de métricas estatísticas para identificar a abordagem que apresenta melhor capacidade preditiva.

---

## Dataset

O conjunto de dados utilizado está disponível no Kaggle:

**Ford Car Price Prediction**

https://www.kaggle.com/datasets/adhurimquku/ford-car-price-prediction

O dataset contém informações sobre veículos Ford anunciados para venda, incluindo atributos como:

- Modelo
- Ano
- Quilometragem
- Tipo de transmissão
- Tipo de combustível
- Consumo (MPG)
- Tamanho do motor
- Taxa (Tax)
- Preço (variável alvo)

---

## Objetivos

Este projeto busca responder às seguintes perguntas:

- É possível prever o preço de venda de um veículo Ford utilizando algoritmos de regressão?
- Qual estratégia de codificação das variáveis categóricas apresenta melhor desempenho?
    - One-Hot Encoding
    - Label Encoding
- Como diferentes métricas de avaliação variam entre os modelos treinados?

---

## Tecnologias utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost
- KaggleHub

---

## Etapas do projeto

### 1. Carregamento dos dados

O dataset é carregado diretamente do Kaggle utilizando a biblioteca `kagglehub`.

---

### 2. Análise Exploratória dos Dados (EDA)

Foram realizadas diversas análises para compreender o comportamento dos dados:

- Estrutura do dataset
- Estatísticas descritivas
- Valores ausentes
- Distribuição do preço
- Correlação entre variáveis numéricas
- Boxplots para análise de variáveis categóricas
- Scatterplots
- Identificação de possíveis inconsistências

---

### 3. Tratamento dos dados

Nesta etapa são realizados:

- Separação entre variáveis independentes e variável alvo;
- Remoção de registros inconsistentes;
- Padronização das variáveis numéricas utilizando `StandardScaler`.

---

### 4. Pré-processamento das variáveis categóricas

O principal experimento deste projeto consiste na comparação entre duas técnicas de codificação.

### Abordagem 1 — One-Hot Encoding

As variáveis:

- model
- transmission
- fuelType

são transformadas em variáveis dummies utilizando:

```python
pd.get_dummies()
```

---

### Abordagem 2 — Label Encoding

As mesmas variáveis são codificadas utilizando:

```python
LabelEncoder()
```

Essa abordagem permite comparar se um modelo utilizando códigos inteiros para categorias apresenta desempenho semelhante ou inferior ao modelo baseado em variáveis dummies.

---

## Modelagem

Inicialmente está sendo utilizado o algoritmo:

- Linear Regression

Posteriormente serão adicionados outros algoritmos para comparação, como:

- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor

---

## Avaliação dos modelos

Os modelos serão comparados utilizando diferentes métricas de regressão.

Entre elas:

- R²
- R² Ajustado
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- Cross Validation

Essas métricas permitem avaliar tanto a capacidade de explicação quanto o erro médio das previsões.

---

## Estrutura do projeto

```
├── data/
├── notebooks/
├── src/
├── images/
├── README.md
└── requirements.txt
```

---

## Resultados

Os resultados serão atualizados conforme o desenvolvimento do projeto.

O objetivo final é comparar:

| Modelo | Pré-processamento | R² | RMSE | MAE | MAPE |
|---------|------------------|----|------|-----|------|
| Linear Regression | One-Hot Encoding | - | - | - | - |
| Linear Regression | Label Encoding | - | - | - | - |
| Decision Tree | One-Hot Encoding | - | - | - | - |
| Decision Tree | Label Encoding | - | - | - | - |
| Random Forest | One-Hot Encoding | - | - | - | - |
| Random Forest | Label Encoding | - | - | - | - |
| XGBoost | One-Hot Encoding | - | - | - | - |

---

## Aprendizados

Durante o desenvolvimento deste projeto são explorados conceitos importantes de Machine Learning, como:

- Análise Exploratória de Dados (EDA)
- Pré-processamento de dados
- Engenharia de atributos
- Codificação de variáveis categóricas
- Padronização de atributos
- Regressão Linear
- Avaliação de modelos
- Validação Cruzada
- Comparação entre algoritmos de regressão

---

## Como executar o projeto

Clone o repositório:

```bash
git clone https://github.com/seu-usuario/nome-do-repositorio.git
```



Execute o arquivo principal do projeto.

---

## Melhorias futuras

- Implementar novos algoritmos de regressão;
- Realizar ajuste de hiperparâmetros;
- Aplicar Grid Search e Random Search;
- Utilizar Pipeline do Scikit-Learn;
- Avaliar importância das variáveis;
- Interpretabilidade do modelo com SHAP;
- Comparar diferentes técnicas de codificação de categorias;
- Publicar o modelo utilizando Streamlit.

---

## Autor

Desenvolvido por **Breno Scaffo** como projeto de estudo em Ciência de Dados e Machine Learning.
