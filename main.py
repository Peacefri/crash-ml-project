
# Project synced with GitHub
# Imports
from road_data import get_road_type  # Function to get road info using OSMnx
from weather_data import get_weather 
from visuals_data import create_visualizations
import pandas as pd


import time


#Load CSV & Clean Column Names

df = pd.read_csv("Austin_crash_report_data.csv", skiprows=1)
df.columns = df.columns.str.strip()

print("Dataset Loaded Successfully")
print(df.head())

# Convert Timestamp & Create Features

df["Crash timestamp (US/Central)"] = pd.to_datetime(
    df["Crash timestamp (US/Central)"],
    format="%Y %b %d %I:%M:%S %p"
)

df["Crash Date"] = df["Crash timestamp (US/Central)"].dt.date
df["Crash Time"] = df["Crash timestamp (US/Central)"].dt.strftime("%H:%M")
df["Crash Hour"] = df["Crash timestamp (US/Central)"].dt.hour
df["Crash Day"] = df["Crash timestamp (US/Central)"].dt.day_name()

#To see all crashes date and time
#print(f"Total accidents processed: {len(df)}")
#print("-" * 25)
#print(df[["Crash Date", "Crash Time"]].to_string(index= False))
print(df[["Crash Date", "Crash Time"]].head())



# Prepare Lists for Enrichment

temps = []
precips = []
is_wet_list = []
highways = []
lanes_list = []


# Loop Through Dataset & Enrich

for i, row in df.iterrows():
    print(f"Processing row {i+1} of {len(df)}")
    
    # Road info
    highway, lanes = get_road_type(row["latitude"], row["longitude"])
    highways.append(highway)
    lanes_list.append(lanes)
    
    # Weather info
    temp, precip = get_weather(
        row["latitude"],
        row["longitude"],
        str(row["Crash Date"]),
        row["Crash Hour"]
    )
    temps.append(temp)
    precips.append(precip)
    is_wet_list.append(precip > 0 if precip is not None else False)
    
    # Prevent API rate limits
    time.sleep(0.5)


# Add Enriched Data to DataFrame
df["Temperature"] = temps
df["Precipitation"] = precips
df["is_wet"] = is_wet_list
df["Highway_Type"] = highways
df["Num_Lanes"] = lanes_list

# Save enriched CSV
df.to_csv("crashes_final_enriched.csv", index=False)
print("enrichment complete!")

# Call visualization script
create_visualizations(df)
