# Trabalho_ECD15_MLOps

## Objetivo:

O objetivo deste trabalho foi aplicar conceitos e práticas de MLOps para desenvolver um pipeline de Machine Learning funcional e automatizado. Além disso, explorar um conjunto de dados real, implementar modelos preditivos e integrar o processo com ferramentas de monitoramento, versionamento e deploy. O foco do projeto foi a construção de um fluxo completo, contemplando desde a preparação dos dados até a entrega do modelo em produção, garantindo rastreabilidade e reprodutibilidade.


## Dataset e Problema:
Loan Default Prediction Dataset
[Acesso](https://www.kaggle.com/datasets/nikhil1e9/loan-default)

Prever inadimplência do cliente para pagamento de empréstimo.

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


2 - Treinamento e Avaliação do Modelo
- Implementação dos modelos Random Forest Classifier e XGBoost e comparação de métricas.
- Utilização do MLflow para rastrear experimentos.


3 - Versionamento e Armazenamento do Modelo
- Registro do modelo no MLflow Model Registry.


4 - Implantação do Modelo
- Construção de uma API com FastAPI ou Flask para servir previsões (mlflow).
- Deploy local.


**5 - Monitoramento e Re-treinamento
- Implementação de monitoramento de drift de dados com Evidently AI.
- Definição de uma estratégia para re-treinamento automático do modelo.


**6 - Conteinerização e Documentação
- Instruções de execução/documentação do pipeline no repositório.



# Execução:

## RandomForestGridSearch

1 - antes de tudo, certificar que tem instalado as tecnologias:
pip install mlflow pandas scikit-learn xgboost matplotlib


2 - para executar o modelo Random Forest Classifier
- em um terminal: python defaultRandomForest.py (ou python3 defaultRandomForest.py)

- em outro terminal: mlflow ui --backend-store-uri sqlite:///mlflowRf.db --port 5000

2.1 - Para fazer o promote do melhor modelo com valor F1 melhor que 0.60
- em um terminal: python defaultRf_promote.py (ou python3 defaultRf_promote.py)


executar o models.py para registro do modelo.


2.2 - Para fazer deploy do modelo em produção, deve:
configurar variável de ambiente:
- MLFLOW_TRACKING_URI_RF --> "sqlite:///mlflowRf.db"
e então rodar o comando:
- mlflow models serve -m "models:/RandomForestGridSearch/Production" --env-manager virtualenv --no-conda --port 8000

## XGBoostGridSearch

3 - para executar o modelo XGBoost
- em um terminal: python defaultXgBoost.py (ou python3 defaultXgBoost.py)

- em outro terminal: mlflow ui --backend-store-uri sqlite:///mlflowXg.db --port 7000

3.1 - Para fazer o promote do melhor modelo com valor F1 melhor que 0.60
- em um terminal: python defaultXg_promote.py (ou python3 defaultXg_promote.py)


executar o models.py para registro do modelo.


3.2 - Para fazer deploy do modelo em produção, deve:
configurar variável de ambiente:
- MLFLOW_TRACKING_URI_XG --> "sqlite:///mlflowXg.db"
e então rodar o comando:
- mlflow models serve -m "models:/XGBoostGridSearch/Production" --env-manager virtualenv --no-conda --port 9000

para teste, usar o payload abaixo:
curl -X POST "http://localhost:8000/invocations" \

     -H "Content-Type: application/json" \

     -d '{

           "instances": [

             {
             "Age": 28.0,
             
             "Income": 9579,
             
             "LoanAmount": 20500,
             
             "CreditScore": 700,
             
             "MonthsEmployed": 70,
             
             "NumCreditLines": 4,
             
             "InterestRate": 15.23,
             
             "LoanTerm": 36,
             
             "DTIRatio": 0.68,
             
             "Education": 1,
             
             "EmploymentType": 1,
             
             "MaritalStatus": 0,
             
             "HasMortgage": 1,
             
             "HasDependents": 1,
             
             "LoanPurpose": 0,
             
             "HasCoSigner": 1: 

             }

           ]

         }'



