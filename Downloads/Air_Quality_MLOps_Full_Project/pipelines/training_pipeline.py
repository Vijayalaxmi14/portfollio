
from zenml import pipeline
from steps.ingest import ingest_data
from steps.preprocess import preprocess_data
from steps.feature_select import feature_selection
from steps.train import train_models
from steps.evaluate import evaluate_models

@pipeline
def training_pipeline():
    df = ingest_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    Xtr, Xte = feature_selection(X_train, X_test, y_train)
    models = train_models(Xtr, y_train)
    evaluate_models(models, Xte, y_test)
