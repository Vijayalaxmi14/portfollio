def classify_aqi(aqi):
    if aqi <= 50:
        return "Good 🟢"
    elif aqi <= 100:
        return "Moderate 🟡"
    elif aqi <= 200:
        return "Unhealthy 🔴"
    else:
        return "Hazardous ☠️"

if __name__ == "__main__":
    predicted_aqi = 180
    alert = classify_aqi(predicted_aqi)
    print(f"Predicted AQI: {predicted_aqi} → {alert}")
