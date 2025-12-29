
from zenml import step
from sklearn.metrics import mean_squared_error, r2_score

@step
def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)
        r2 = r2_score(y_test, preds)
        results[name] = {"RMSE": rmse, "R2": r2}
        print(name, results[name])
    return results
