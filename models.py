import mlflow
from mlflow.tracking import MlflowClient

clientRf = MlflowClient(tracking_uri="sqlite:///mlflowRf.db")
clientXg = MlflowClient(tracking_uri="sqlite:///mlflowXg.db")

# Verifica o modelo RandomForestGridSearch registrado
modelsRf = clientRf.search_registered_models()
for model in modelsRf:
    print(f"Modelo: {model.name}, Local: {model.latest_versions}")

print("/////////////////////////////")

# Verifica o modelo xgBoostGridSearch registrado
modelsXg = clientXg.search_registered_models()
for model in modelsXg:
    print(f"Modelo: {model.name}, Local: {model.latest_versions}")



model_uri = "file:///C:/Users/Everton Vaszelewski/Downloads/Trabalho_ECD15_MLOps-master/mlruns/1/b7096dbb02f145e78ff54af0ffe295b5/artifacts/random_forest_model"
mlflow.register_model(model_uri, "RandomForestGridSearch")

model_uriXG = "file:///C:/Users/Everton Vaszelewski/Downloads/Trabalho_ECD15_MLOps-master/mlruns/1/6f93e7e89ad2497090694d8244befd59/artifacts/xgboost_grid_search"
mlflow.register_model(model_uriXG, "XGBoostGridSearch")
