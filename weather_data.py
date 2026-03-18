import requests
import json
import os

WEATHER_CACHE_FILE = "weather_cache.json"

# Load persistent cache from disk if it exists
if os.path.exists(WEATHER_CACHE_FILE):
    with open(WEATHER_CACHE_FILE, "r") as f:
        weather_cache = json.load(f)
    print(f"Weather cache loaded — {len(weather_cache)} entries found")
else:
    weather_cache = {}


def save_cache():
    """Save the in-memory cache to disk so it persists across runs."""
    with open(WEATHER_CACHE_FILE, "w") as f:
        json.dump(weather_cache, f)


def get_weather(lat, lon, date, hour):
    """
    Returns weather conditions for a given location and time
    using the Open-Meteo Historical API.

    Returns: (temp, precip, windspeed, visibility, weathercode)
    """
    lat = round(lat, 2)
    lon = round(lon, 2)
    hour = int(hour)

    # String key for JSON-serializable cache
    key = f"{lat}_{lon}_{date}_{hour}"

    if key in weather_cache:
        return tuple(weather_cache[key])

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date,
        "end_date": date,
        "hourly": ",".join([
            "temperature_2m",
            "precipitation",
            "windspeed_10m",       # High wind = crash risk
            "visibility",          # Low visibility = crash risk
            "weathercode",         # Coded condition (fog, snow, storm, etc.)
            "cloudcover"           # General sky condition
        ]),
        "timezone": "America/Chicago"  # Austin is US Central — explicit is safer than "auto"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "hourly" not in data:
            print(f"  Weather API: no hourly data returned for {date} at ({lat}, {lon})")
            return None, None, None, None, None

        times = data["hourly"]["time"]
        target_time = f"{date}T{str(hour).zfill(2)}:00"

        if target_time not in times:
            print(f"  Weather API: target time {target_time} not found in response")
            return None, None, None, None, None

        index = times.index(target_time)
        hourly = data["hourly"]

        def safe_get(field):
            """Return None if field missing or value is null."""
            val = hourly.get(field, [None])
            if index < len(val):
                return val[index]
            return None

        temp        = safe_get("temperature_2m")
        precip      = safe_get("precipitation")
        windspeed   = safe_get("windspeed_10m")
        visibility  = safe_get("visibility")
        weathercode = safe_get("weathercode")

        result = (temp, precip, windspeed, visibility, weathercode)

        # Save to cache and persist to disk
        weather_cache[key] = list(result)
        save_cache()

        return result

    except requests.exceptions.Timeout:
        print(f"  Weather API: timeout for ({lat}, {lon}) on {date}")
    except requests.exceptions.HTTPError as e:
        print(f"  Weather API: HTTP error {e} for ({lat}, {lon}) on {date}")
    except requests.exceptions.RequestException as e:
        print(f"  Weather API: request failed — {e}")
    except (KeyError, ValueError) as e:
        print(f"  Weather API: data parsing error — {e}")

    return None, None, None, None, None


def decode_weathercode(code):
    """
    Translate WMO weather code into a human-readable category.
    Useful for feature engineering and visualizations.
    """
    if code is None:
        return "Unknown"
    if code == 0:
        return "Clear"
    elif code in range(1, 4):
        return "Partly Cloudy"
    elif code in range(10, 20):
        return "Foggy"
    elif code in range(20, 30):
        return "Drizzle/Rain"
    elif code in range(51, 68):
        return "Rain"
    elif code in range(71, 78):
        return "Snow"
    elif code in range(80, 83):
        return "Rain Showers"
    elif code in range(95, 100):
        return "Thunderstorm"
    else:
        return "Other"