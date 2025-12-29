import pandas as pd
import mlflow
from sklearn.feature_selection import SelectKBest, f_regression

mlflow.set_experiment("Feature_Selection")

df = pd.read_csv("data/processed/air_quality_processed.csv")
X = df.drop("AQI", axis=1)
y = df["AQI"]

with mlflow.start_run():
    selector = SelectKBest(score_func=f_regression, k=3)
    X_new = selector.fit_transform(X, y)

    selected_features = X.columns[selector.get_support()]
    mlflow.log_param("selected_features", list(selected_features))
