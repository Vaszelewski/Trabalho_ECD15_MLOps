# Trabalho_ECD15_MLOps

## Objetivo:

O objetivo deste trabalho foi aplicar conceitos e práticas de MLOps para desenvolver um pipeline de Machine Learning funcional e automatizado. Além disso, explorar um conjunto de dados real, implementar modelos preditivos e integrar o processo com ferramentas de monitoramento, versionamento e deploy. O foco do projeto foi a construção de um fluxo completo, contemplando desde a preparação dos dados até a entrega do modelo em produção, garantindo rastreabilidade e reprodutibilidade.


## Dataset e Problema:
Loan Default Prediction Dataset
[Acesso](https://www.kaggle.com/datasets/nikhil1e9/loan-default)

Prever prever inadimplência do cliente para pagamento de empréstimo.

## Ferramentas:

- Linguagem: Python 3.10
- MLflow (para rastreamento e versionamento de modelos)
- ***Evidently AI (para monitoramento de drift)
- FastAPI/Flask (para disponibilização do modelo via API, (ex. usando Mlflow)
- GitHub (para controle de versão)

## Etapas do Projeto (MLOps Pipeline)
1 - Exploração e Pré-processamento dos Dados
- Análise exploratória e tratamento de valores ausentes.
- Normalização/Padronização dos dados quando necessário.


## Execução:

pip install mlflow pandas scikit-learn matplotlib

mlflow ui --backend-store-uri sqlite:///mlflow.db

