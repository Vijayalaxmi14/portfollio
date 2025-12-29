import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib
import numpy as np

df = pd.read_csv("data/processed/air_quality_processed.csv")
X = df.drop("AQI", axis=1)
y = df["AQI"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(),
    "XGBoost": XGBRegressor()
}

best_rmse = np.inf
best_model = None

mlflow.set_experiment("Model_Training")

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
            joblib.dump(best_model, "models/best_model.pkl")
