
def check_alert(value):
    if value > 150:
        return "⚠️ Unhealthy Air Quality Alert"
    return "Air Quality Normal"
