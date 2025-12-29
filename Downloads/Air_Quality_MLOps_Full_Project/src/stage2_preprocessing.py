import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def preprocess():
    df = pd.read_csv("data/raw/air_quality.csv")

    X = df.drop("AQI", axis=1)
    y = df["AQI"]

    pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    X_scaled = pipeline.fit_transform(X)
    processed = pd.DataFrame(X_scaled, columns=X.columns)
    processed["AQI"] = y

    processed.to_csv("data/processed/air_quality_processed.csv", index=False)

if __name__ == "__main__":
    preprocess()
