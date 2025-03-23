import pandas as pd
import mlflow
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
from mlflow.models.signature import infer_signature

mlflow.set_tracking_uri("sqlite:///mlflowXg.db")
mlflow.set_experiment("loandefaultpredXg_experiment")

def preprocess_data(df):
    df.drop(columns=["LoanID"], inplace=True, errors="ignore")
    df.replace({"Yes": 1, "No": 0}, inplace=True)
    df = df.infer_objects(copy=False)

    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype("float64")
    
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str)
        df[col] = LabelEncoder().fit_transform(df[col])

    df.fillna(0, inplace=True)

    X = df.drop(columns=["Default"])
    y = df["Default"]

    mlflow_dataset = mlflow.data.from_pandas(df, targets="Default")

    return X, y, mlflow_dataset

def load_data(dataset_name):
    df = pd.read_csv(dataset_name)
    return df

def train_xgboost_grid_search(X, y, dataset_name, mlflow_dataset):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    param_grid = {
        "n_estimators": [50, 100], #"n_estimators": [50, 100],
        "max_depth": [10, 20], #"max_depth": [3, 5],
        "learning_rate": [0.01, 0.1],
    }
    
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    grid_search = GridSearchCV(xgb, param_grid, scoring='accuracy', cv=3)
    grid_search.fit(X_train, y_train)

    for params in (dict(zip(param_grid.keys(), values)) for values in
                   [(n, d, s) for n in param_grid["n_estimators"]
                            for d in param_grid["max_depth"]
                            for s in param_grid["learning_rate"]]):
        
    
        xgb.set_params(**params)
        xgb.fit(X_train, y_train)
        y_test_pred = xgb.predict(X_test)
        accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred)
        recall = recall_score(y_test, y_test_pred)
        with mlflow.start_run(run_name=f"XGB_{params['n_estimators']}_{params['max_depth']}_{params['learning_rate']}"):
            mlflow.log_input(mlflow_dataset, context="training")
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.set_tag("dataset_used", dataset_name)
        
            signature = infer_signature(X_train, y_test_pred)
            model_info = mlflow.sklearn.log_model(xgb, "xgboost_grid_search",
                                     signature=signature,
                                     input_example=X_train,
                                     registered_model_name="XGBoostGridSearch")
        
            loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
            predictions = loaded_model.predict(X_test)
            result = pd.DataFrame(X_test, columns=X.columns.values)
            result["label"] = y_test.values
            result["predictions"] = predictions

            mlflow.evaluate(
                data=result,
                targets="label",
                predictions="predictions",
                model_type="classifier",
            )

            print(result[:5])


    return xgb

def main():
    dataset_name = "LoanDefaultPred.csv"
    df = load_data(dataset_name)
    X, y, mlflow_dataset = preprocess_data(df)
    xgb_model = train_xgboost_grid_search(X, y, dataset_name, mlflow_dataset)

if __name__ == "__main__":
    main()