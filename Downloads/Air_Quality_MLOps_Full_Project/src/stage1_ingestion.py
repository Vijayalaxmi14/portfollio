import pandas as pd
import os

def ingest_data():
    os.makedirs("data/raw", exist_ok=True)

    # Sample AQI dataset
    data = {
        "PM2.5": [55, 40, 70, 90, 110],
        "PM10": [80, 60, 100, 130, 160],
        "NO2": [30, 25, 40, 50, 65],
        "SO2": [10, 12, 15, 20, 25],
        "CO": [0.8, 0.6, 1.2, 1.6, 2.0],
        "AQI": [120, 90, 160, 200, 260]
    }

    df = pd.DataFrame(data)
    df.to_csv("data/raw/city_day.csv", index=False)

if __name__ == "__main__":
    ingest_data()
