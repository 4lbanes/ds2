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
- `data/processed/treino_limpo.csv`: base limpa produzida no Acompanhamento 1 e usada no treino dos modelos.
- `data/processed/teste_publico_limpo.csv`: base limpa produzida no Acompanhamento 1 e usada para validar o `pipeline.py`.
- `acompanhamentos/acomp2/metricas_preliminares.csv`: metricas dos modelos treinados neste acompanhamento.
- `acompanhamentos/acomp2/predicoes_teste_publico.csv`: predicoes geradas para a base limpa de teste publico.
- `acompanhamentos/acomp2/modelo_acomp2.joblib`: artefato do melhor modelo preliminar salvo para o `pipeline.py`.

O Acompanhamento 2 nao treina diretamente sobre os CSVs brutos. Ele consome os
datasets limpos em `data/processed/`, preservando o handoff entre limpeza,
modelagem e predicao.

## Arquivos principais

- `acomp_2_pipeline_modelos_preliminares.ipynb`: notebook principal para apresentacao em sala.
- `treino_modelos.py`: treina Ridge, Random Forest e Gradient Boosting, compara metricas e salva o melhor modelo.
- `pipeline.py`: carrega `modelo_acomp2.joblib` e expoe `predict(input_csv)`, retornando apenas um `np.ndarray` de predicoes.
- `metricas_preliminares.csv`: tabela com RMSLE, RMSE, MAE, R2 e tempo de treino.
- `predicoes_teste_publico.csv`: predicoes geradas a partir de `data/processed/teste_publico_limpo.csv`.

## Modelos testados

Todos os modelos usam o mesmo pre-processamento:

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

- le `data/processed/treino_limpo.csv`;
- treina e valida os modelos preliminares;
- salva `modelo_acomp2.joblib`;
- atualiza `metricas_preliminares.csv`;
- gera `predicoes_teste_publico.csv` usando `data/processed/teste_publico_limpo.csv`.

Para demonstrar o `pipeline.py` rodando:

```bash
.venv/bin/python acompanhamentos/acomp2/pipeline.py data/processed/teste_publico_limpo.csv
```

## Ordem sugerida de apresentacao

1. Abrir `acomp_2_pipeline_modelos_preliminares.ipynb`.
2. Mostrar que os dados consumidos vêm de `data/processed/`.
3. Mostrar a tabela de metricas preliminares.
4. Executar o `pipeline.py` com `data/processed/teste_publico_limpo.csv`.
5. Mostrar que a saida tem uma predicao por linha e nao inclui coluna de ID.
