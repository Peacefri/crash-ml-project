

# Imports
from road_data import get_road_type
from weather_data import get_weather
from visuals_data import create_visualizations
from visuals_data import create_crash_heatmap
import pandas as pd
import time

INPUT_FILE = "Austin_crash_report_data.csv"
OUTPUT_FILE = "crashes_final_enriched.csv"


def load_data():
    df = pd.read_csv(INPUT_FILE, skiprows=1)
    df.columns = df.columns.str.strip()
    print("Dataset Loaded Successfully")
    return df


def create_time_features(df):
    df["Crash timestamp (US/Central)"] = pd.to_datetime(
        df["Crash timestamp (US/Central)"],
        format="%Y %b %d %I:%M:%S %p"
    )

    df["Crash Date"] = df["Crash timestamp (US/Central)"].dt.date
    df["Crash Hour"] = df["Crash timestamp (US/Central)"].dt.hour
    df["Crash Day"] = df["Crash timestamp (US/Central)"].dt.day_name()

    return df


def enrich_data(df):

    temps = []
    precips = []
    highways = []
    lanes_list = []
    is_wet_list = []

    for i, row in df.iterrows():

        highway, lanes = get_road_type(row["latitude"], row["longitude"])
        temp, precip = get_weather(
            row["latitude"],
            row["longitude"],
            str(row["Crash Date"]),
            row["Crash Hour"]
        )

        temps.append(temp)
        precips.append(precip)
        highways.append(highway)
        lanes_list.append(lanes)
        is_wet_list.append(precip > 0 if precip else False)

        time.sleep(0.5)

    df["Temperature"] = temps
    df["Precipitation"] = precips
    df["is_wet"] = is_wet_list
    df["Highway_Type"] = highways
    df["Num_Lanes"] = lanes_list

    return df


def main():

    df = load_data()
    df = create_time_features(df)
    df = enrich_data(df)

    df.to_csv(OUTPUT_FILE, index=False)
    print("Enrichment Complete!")

    create_visualizations(df)
    create_crash_heatmap(df)


if __name__ == "__main__":
    main()
