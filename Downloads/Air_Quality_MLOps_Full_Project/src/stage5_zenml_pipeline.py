from zenml import pipeline, step
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import numpy as np
import mlflow

# ---------------- STEP 1: Load Data ----------------
@step
def load_data():
    df = pd.read_csv("data/processed/air_quality_processed.csv")
    X = df.drop("AQI", axis=1)
    y = df["AQI"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- STEP 2: Train & Compare Models ----------------
@step
def train_models(data):
    X_train, X_test, y_train, y_test = data

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42)
    }

    best_model = None
    best_rmse = np.inf

    mlflow.set_experiment("ZenML_Training_Pipeline")

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            rmse = mean_squared_error(y_test, preds, squared=False)
            r2 = r2_score(y_test, preds)

            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R2", r2)

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model

    return best_model

# ---------------- STEP 3: Save Best Model ----------------
@step
def save_model(model):
    joblib.dump(model, "models/best_model.pkl")
    return "Model saved successfully"

# ---------------- PIPELINE ----------------
@pipeline
def air_quality_training_pipeline(load_data, train_models, save_model):
    data = load_data()
    model = train_models(data)
    save_model(model)

# ---------------- RUN PIPELINE ----------------
if __name__ == "__main__":
    air_quality_training_pipeline(
        load_data=load_data,
        train_models=train_models,
        save_model=save_model
    )()
