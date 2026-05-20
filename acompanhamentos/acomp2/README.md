# Acompanhamento 2

Entrega referente ao segundo checkpoint do projeto de Machine Learning para
previsao de precos de imoveis.

## Objetivo do acompanhamento

O PDF do projeto define o Acompanhamento 2 como a apresentacao de:

- estrutura do `pipeline.py` rodando;
- resultados preliminares dos primeiros modelos;
- uso de um fluxo reprodutivel de pre-processamento e predicao.

Esta pasta atende esse escopo com um notebook de apresentacao, um script de treino,
um `pipeline.py` executavel e os artefatos gerados.

## Fluxo correto dos dados

Os dados estao separados por responsabilidade:

- `data/treino.csv`: base original de treino, mantida intacta.
- `data/teste_publico.csv`: base original de teste publico, mantida intacta.
- `data/processed/treino_limpo.csv`: evidencia da limpeza documentada no Acompanhamento 1, nao e mais a entrada do Acompanhamento 2.
- `data/processed/teste_publico_limpo.csv`: evidencia da limpeza documentada no Acompanhamento 1, nao e mais a entrada do `pipeline.py`.
- `acompanhamentos/acomp2/metricas_preliminares.csv`: metricas dos modelos treinados neste acompanhamento.
- `acompanhamentos/acomp2/predicoes_teste_publico.csv`: predicoes geradas para a base bruta de teste publico.
- `acompanhamentos/acomp2/modelo_acomp2.joblib`: artefato do melhor modelo preliminar salvo para o `pipeline.py`.

O Acompanhamento 2 deve partir dos CSVs brutos. A limpeza foi migrada para dentro
do `pipeline.py`, por meio do transformador `HousePricesCleaner`, e fica como a
primeira etapa do `sklearn.Pipeline` salvo em `modelo_acomp2.joblib`. Isso atende
ao PDF: o script de predicao consegue receber um CSV novo e sujo, aplicar os
tratamentos, processar as variaveis e retornar apenas as predicoes.

## Arquivos principais

- `acomp_2_pipeline_modelos_preliminares.ipynb`: notebook principal para apresentacao em sala.
- `treino_modelos.py`: treina Ridge, Random Forest e Gradient Boosting, compara metricas e salva o melhor modelo.
- `pipeline.py`: carrega `modelo_acomp2.joblib` e expoe `predict(input_csv)`, retornando apenas um `np.ndarray` de predicoes.
- `metricas_preliminares.csv`: tabela com RMSLE, RMSE, MAE, R2 e tempo de treino.
- `predicoes_teste_publico.csv`: predicoes geradas a partir de `data/teste_publico.csv`.

## Modelos testados

Todos os modelos usam o mesmo pre-processamento:

- limpeza inicial com `HousePricesCleaner`, replicando a estrategia do Acompanhamento 1;
- imputacao mediana para variaveis numericas;
- imputacao com `Ausente` para variaveis categoricas;
- `StandardScaler` nas variaveis numericas;
- `OneHotEncoder(handle_unknown="ignore")` nas variaveis categoricas;
- `TransformedTargetRegressor` para treinar com `log1p(SalePrice)` e devolver valores em dolares com `expm1`.

Modelos preliminares avaliados:

- Ridge;
- Random Forest;
- Gradient Boosting.

## Como executar

A partir da raiz do projeto:

```bash
.venv/bin/python acompanhamentos/acomp2/treino_modelos.py
```

Esse comando:

- le `data/treino.csv`;
- ajusta a limpeza apenas com os dados de treino de cada divisao de validacao;
- treina e valida os modelos preliminares;
- salva `modelo_acomp2.joblib`;
- atualiza `metricas_preliminares.csv`;
- gera `predicoes_teste_publico.csv` usando `data/teste_publico.csv`.

Para demonstrar o `pipeline.py` rodando:

```bash
.venv/bin/python acompanhamentos/acomp2/pipeline.py data/teste_publico.csv
```

## Ordem sugerida de apresentacao

1. Abrir `acomp_2_pipeline_modelos_preliminares.ipynb`.
2. Mostrar que o treino e o `pipeline.py` recebem os CSVs brutos em `data/`.
3. Mostrar a tabela de metricas preliminares.
4. Executar o `pipeline.py` com `data/teste_publico.csv`.
5. Mostrar que a saida tem uma predicao por linha e nao inclui coluna de ID.
