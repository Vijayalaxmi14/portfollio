
# Air Quality Forecasting & Alert System using MLOps (ZenML)

## Features
- End-to-end ZenML pipeline (5 stages)
- Feature selection comparison
- Model comparison (Linear Regression, Random Forest, XGBoost)
- Metrics tracking (RMSE, R2)
- Streamlit dashboard
- Alert system for poor air quality
- Dockerized setup

## Run Instructions
pip install -r requirements.txt
zenml init
python run_pipeline.py
streamlit run dashboard/app.py
