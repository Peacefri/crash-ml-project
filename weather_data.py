# Weather API Function
import requests

weather_cache = {}

def get_weather(lat, lon, date, hour):
    """
    Returns temperature and precipitation
    for a given location and time using Open-Meteo Historical API
    """

    lat = round(lat, 2)
    lon = round(lon, 2)
    hour = int(hour)

    key = (lat, lon, date, hour)

    if key in weather_cache:
        return weather_cache[key]

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date,
        "end_date": date,
        "hourly": "temperature_2m,precipitation",
        "timezone": "auto"
    }

    try:

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        if "hourly" not in data:
            return None, None

        times = data["hourly"]["time"]
        target_time = f"{date}T{str(hour).zfill(2)}:00"

        if target_time in times:

            index = times.index(target_time)

            temp = data["hourly"]["temperature_2m"][index]
            precip = data["hourly"]["precipitation"][index]

            weather_cache[key] = (temp, precip)

            return temp, precip

    except requests.exceptions.RequestException:
        pass

    return None, None
