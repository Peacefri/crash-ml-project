# Weather API Function
import requests

def get_weather(lat, lon, date, hour):
    """
    Returns temperature and precipitation for a given location and time
    using Open-Meteo Historical Weather API
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date,
        "end_date": date,
        "hourly": "temperature_2m,precipitation",
        "timezone": "auto"
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        times = data["hourly"]["time"]
        target_time = f"{date}T{str(hour).zfill(2)}:00"
        
        if target_time in times:
            index = times.index(target_time)
            temp = data["hourly"]["temperature_2m"][index]
            precip = data["hourly"]["precipitation"][index]
            return temp, precip
    return None, None
