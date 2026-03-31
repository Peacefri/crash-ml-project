# ============================================================
# main.py — Austin Crash Safety Prediction System
# Phase 1: Data Collection & Enrichment
# ============================================================

import pandas as pd
import time
import os

from road_data import get_road_type
from weather_data import get_weather, decode_weathercode
from visuals_data import create_visualizations, create_crash_heatmap
from aadt_data import get_aadt


# ── File paths ───────────────────────────────────────────────
INPUT_FILE      = "crashes_working.csv"
OUTPUT_FILE     = "crashes_final_enriched.csv"
CHECKPOINT_FILE = "crashes_checkpoint.csv"


# ── 1. Load Data ─────────────────────────────────────────────
def load_data():
    """
    Load the crash CSV with manually defined column names
    because MySQL exports do not include headers.
    Change nrows=500 to nrows=None when ready for the full run.
    """
    cols = [
        'ID', 'Crash ID', 'crash_fatal_fl', 'case_id', 'rpt_block_num',
        'rpt_street_name', 'rpt_street_sfx', 'crash_speed_limit',
        'road_constr_zone_fl', 'latitude', 'longitude', 'crash_sev_id',
        'sus_serious_injry_cnt', 'nonincap_injry_cnt', 'poss_injry_cnt',
        'non_injry_cnt', 'unkn_injry_cnt', 'tot_injry_cnt', 'death_cnt',
        'units_involved', 'point', 'motor_vehicle_death_count',
        'motor_vehicle_serious_injury_count', 'bicycle_death_count',
        'bicycle_serious_injury_count', 'pedestrian_death_count',
        'pedestrian_serious_injury_count', 'motorcycle_death_count',
        'motorcycle_serious_injury_count', 'other_death_count',
        'other_serious_injury_count', 'onsys_fl', 'private_dr_fl',
        'micromobility_serious_injury_count', 'micromobility_death_count',
        'Crash timestamp (US/Central)', 'Crash timestamp',
        'Is deleted', 'Is temporary record', 'Law enforcement fatality count',
        'Reported street prefix', 'Estimated Maximum Comprehensive Cost',
        'Estimated Total Comprehensive Cost', 'Location ID',
        'Location group', 'Address', 'Collision type'
    ]

    df = pd.read_csv(
        INPUT_FILE,
        header=0,
        names=cols,
        low_memory=False,
        nrows=500          # Remove this line for the full overnight run
    )
    df.columns = df.columns.str.strip()
    print(f"Dataset loaded successfully — {len(df)} rows, {len(df.columns)} columns")
    return df


# ── 2. Time Features ─────────────────────────────────────────
def create_time_features(df):
    """
    Parse the crash timestamp and extract useful time-based
    features for the prediction model.
    """
    df["Crash timestamp (US/Central)"] = pd.to_datetime(
        df["Crash timestamp (US/Central)"],
        format="%Y %b %d %I:%M:%S %p",
        errors="coerce"
    )

    bad_dates = df["Crash timestamp (US/Central)"].isna().sum()
    if bad_dates > 0:
        print(f"  Warning: {bad_dates} timestamps could not be parsed")

    df["Crash Date"]  = df["Crash timestamp (US/Central)"].dt.date
    df["Crash Hour"]  = df["Crash timestamp (US/Central)"].dt.hour
    df["Crash Day"]   = df["Crash timestamp (US/Central)"].dt.day_name()
    df["Crash Month"] = df["Crash timestamp (US/Central)"].dt.month
    df["Crash Year"]  = df["Crash timestamp (US/Central)"].dt.year
    df["Is_Weekend"]  = df["Crash Day"].isin(["Saturday", "Sunday"])

    print("  Time features created: Date, Hour, Day, Month, Year, Is_Weekend")
    return df


# ── 3. Severity Label ────────────────────────────────────────
def create_severity_label(df):
    """
    Add human-readable severity label and binary Is_Severe
    target column for the ML model.

    Austin severity codes:
        0 = Unknown
        1 = Incapacitating Injury
        2 = Non-Incapacitating Injury
        3 = Possible Injury
        4 = Killed
        5 = Not Injured
    """
    severity_labels = {
        0: "Unknown",
        1: "Incapacitating Injury",
        2: "Non-Incapacitating Injury",
        3: "Possible Injury",
        4: "Killed",
        5: "Not Injured"
    }

    df["Severity_Label"] = df["crash_sev_id"].map(severity_labels).fillna("Unknown")

    # Binary target: severe = Incapacitating (1) or Killed (4)
    df["Is_Severe"] = df["crash_sev_id"].isin([0, 1, 4]).astype(int)

    print("  Severity labels and Is_Severe target column created")
    return df


# ── 4. Enrich Data ───────────────────────────────────────────
def enrich_data(df):
    """
    Loop through every crash row and call road, weather,
    and AADT APIs to attach enriched context to each record.
    """

    # ── Checkpoint Recovery ───────────────────────────────────
    if os.path.exists(CHECKPOINT_FILE):
        print(f"\n  Checkpoint found — resuming from saved progress...")
        checkpoint_df   = pd.read_csv(CHECKPOINT_FILE)
        processed_count = len(checkpoint_df)
        print(f"  Skipping first {processed_count} already-processed rows\n")
    else:
        processed_count = 0

    # ── Road output lists ─────────────────────────────────────
    highways             = []
    highway_labels       = []
    road_names           = []
    lanes_list           = []
    speed_limits         = []
    road_risks           = []
    is_intersections     = []
    intersection_degrees = []
    curvatures           = []

    # ── AADT output lists ─────────────────────────────────────
    aadt_values    = []
    aadt_roads     = []
    aadt_distances = []

    # ── Weather output lists ──────────────────────────────────
    temps        = []
    precips      = []
    windspeeds   = []
    visibilities = []
    weathercodes = []
    is_wet_list  = []

    total = len(df)

    # ── Row Loop ─────────────────────────────────────────────
    for i, row in df.iterrows():

        # Skip rows already processed in a previous run
        if i < processed_count:
            continue

        # ── Validate coordinates ──────────────────────────────
        lat = row.get("latitude")
        lon = row.get("longitude")

        if pd.isna(lat) or pd.isna(lon):
            print(f"  Row {i}: Missing coordinates — skipping API calls")
            highways.append(None)
            highway_labels.append(None)
            road_names.append(None)
            lanes_list.append(None)
            speed_limits.append(None)
            road_risks.append(None)
            is_intersections.append(None)
            intersection_degrees.append(None)
            curvatures.append(None)
            aadt_values.append(None)
            aadt_roads.append(None)
            aadt_distances.append(None)
            temps.append(None)
            precips.append(None)
            windspeeds.append(None)
            visibilities.append(None)
            weathercodes.append(None)
            is_wet_list.append(None)
            continue

        # ── Road Data ─────────────────────────────────────────
        try:
            (highway, highway_label, lanes, road_risk, speed,
             is_intersection, intersection_degree,
             curvature, road_name) = get_road_type(lat, lon)
        except Exception as e:
            print(f"  Row {i}: Road data failed — {e}")
            (highway, highway_label, lanes, road_risk, speed,
             is_intersection, intersection_degree,
             curvature, road_name) = (None, None, None, None, None,
                                      None, None, None, None)

        highways.append(highway)
        highway_labels.append(highway_label)
        road_names.append(road_name)
        lanes_list.append(lanes)
        speed_limits.append(speed)
        road_risks.append(road_risk)
        is_intersections.append(is_intersection)
        intersection_degrees.append(intersection_degree)
        curvatures.append(curvature)

        # ── AADT Data ─────────────────────────────────────────
        try:
            crash_year = row.get("Crash Year")
            if pd.isna(crash_year):
                crash_year = pd.to_datetime(
                    row["Crash timestamp (US/Central)"], errors="coerce"
                ).year
            aadt_val, aadt_road, aadt_dist, _ = get_aadt(
                lat, lon, int(crash_year)
            )
        except Exception as e:
            print(f"  Row {i}: AADT lookup failed — {e}")
            aadt_val, aadt_road, aadt_dist = None, None, None

        aadt_values.append(aadt_val)
        aadt_roads.append(aadt_road)
        aadt_distances.append(aadt_dist)

        # ── Weather Data ──────────────────────────────────────
        try:
            temp, precip, windspeed, visibility, weathercode = get_weather(
                lat,
                lon,
                str(row["Crash Date"]),
                row["Crash Hour"]
            )
        except Exception as e:
            print(f"  Row {i}: Weather data failed — {e}")
            temp, precip, windspeed, visibility, weathercode = (
                None, None, None, None, None
            )

        temps.append(temp)
        precips.append(precip)
        windspeeds.append(windspeed)
        visibilities.append(visibility)
        weathercodes.append(weathercode)
        is_wet_list.append(precip > 0 if precip is not None else None)

        # ── Progress update every 500 rows ────────────────────
        if (i + 1) % 500 == 0 or (i + 1) == total:
            print(f"  Progress: {i + 1}/{total} rows processed...")

        # ── Save checkpoint every 500 rows ────────────────────
        if (i + 1) % 500 == 0:
            _save_checkpoint(
                df, i,
                highways, highway_labels, road_names,
                lanes_list, speed_limits, road_risks,
                is_intersections, intersection_degrees, curvatures,
                aadt_values, aadt_roads, aadt_distances,
                temps, precips, windspeeds, visibilities,
                weathercodes, is_wet_list
            )

        # Sleep every 10 rows to avoid API rate limits
        if i % 10 == 0:
            time.sleep(0.5)

    # ── Attach all enriched columns to dataframe ─────────────

    # Road columns
    df["Highway_Type"]         = highways
    df["Road_Type_Label"]      = highway_labels
    df["Road_Name"]            = road_names
    df["Num_Lanes"]            = lanes_list
    df["Speed_Limit"]          = speed_limits
    df["Road_Risk_Level"]      = road_risks
    df["Is_Intersection"]      = is_intersections
    df["Intersection_Degree"]  = intersection_degrees
    df["Road_Curvature"]       = curvatures

    # AADT columns
    df["AADT"]                 = aadt_values
    df["AADT_Station_Road"]    = aadt_roads
    df["AADT_Distance_km"]     = aadt_distances

    # Weather columns
    df["Temperature"]          = temps
    df["Precipitation"]        = precips
    df["Windspeed"]            = windspeeds
    df["Visibility"]           = visibilities
    df["Weather_Code"]         = weathercodes
    df["Weather_Condition"]    = df["Weather_Code"].apply(decode_weathercode)
    df["is_wet"]               = is_wet_list

    return df


# ── Checkpoint Helper ────────────────────────────────────────
def _save_checkpoint(df, i,
                     highways, highway_labels, road_names,
                     lanes_list, speed_limits, road_risks,
                     is_intersections, intersection_degrees, curvatures,
                     aadt_values, aadt_roads, aadt_distances,
                     temps, precips, windspeeds, visibilities,
                     weathercodes, is_wet_list):
    """
    Save a partial copy of the enriched dataframe to disk.
    Allows the run to resume from this point if interrupted.
    """
    partial = df.iloc[:i + 1].copy()
    n = len(partial)

    def pad(lst):
        return (lst + [None] * n)[:n]

    partial["Highway_Type"]        = pad(highways)
    partial["Road_Type_Label"]     = pad(highway_labels)
    partial["Road_Name"]           = pad(road_names)
    partial["Num_Lanes"]           = pad(lanes_list)
    partial["Speed_Limit"]         = pad(speed_limits)
    partial["Road_Risk_Level"]     = pad(road_risks)
    partial["Is_Intersection"]     = pad(is_intersections)
    partial["Intersection_Degree"] = pad(intersection_degrees)
    partial["Road_Curvature"]      = pad(curvatures)
    partial["AADT"]                = pad(aadt_values)
    partial["AADT_Station_Road"]   = pad(aadt_roads)
    partial["AADT_Distance_km"]    = pad(aadt_distances)
    partial["Temperature"]         = pad(temps)
    partial["Precipitation"]       = pad(precips)
    partial["Windspeed"]           = pad(windspeeds)
    partial["Visibility"]          = pad(visibilities)
    partial["Weather_Code"]        = pad(weathercodes)
    partial["is_wet"]              = pad(is_wet_list)

    partial.to_csv(CHECKPOINT_FILE, index=False)
    print(f"  Checkpoint saved at row {i + 1}")


# ── 5. Main ──────────────────────────────────────────────────
def main():

    print("=" * 55)
    print("  Austin Crash Safety System — Phase 1 Enrichment")
    print("=" * 55)

    print("\n[Step 1] Loading data...")
    df = load_data()

    print("\n[Step 2] Creating time features...")
    df = create_time_features(df)

    print("\n[Step 3] Creating severity labels...")
    df = create_severity_label(df)

    print("\n[Step 4] Enriching data (road + AADT + weather APIs)...")
    print("  This may take a few minutes — progress updates every 500 rows\n")
    df = enrich_data(df)

    print(f"\n[Step 5] Saving enriched dataset to '{OUTPUT_FILE}'...")
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"  Saved — {len(df)} rows, {len(df.columns)} columns")

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("  Checkpoint file removed")

    print("\n[Step 6] Generating visualizations...")
    create_visualizations(df)
    create_crash_heatmap(df)

    print("\n" + "=" * 55)
    print("  Phase 1 Complete!")
    print(f"  Output file  : {OUTPUT_FILE}")
    print(f"  Total rows   : {len(df)}")
    print(f"  Total columns: {len(df.columns)}")
    print()
    print("  Road columns    : Highway_Type, Road_Type_Label, Road_Name,")
    print("                    Num_Lanes, Speed_Limit, Road_Risk_Level,")
    print("                    Is_Intersection, Intersection_Degree, Road_Curvature")
    print("  AADT columns    : AADT, AADT_Station_Road, AADT_Distance_km")
    print("  Weather columns : Temperature, Precipitation, Windspeed,")
    print("                    Visibility, Weather_Code, Weather_Condition, is_wet")
    print("  Time columns    : Crash Date, Hour, Day, Month, Year, Is_Weekend")
    print("  Severity columns: Severity_Label, Is_Severe")
    print()
    print("  Charts saved : see .png files in this folder")
    print("  Maps saved   : see .html files in this folder")
    print("=" * 55)


if __name__ == "__main__":
    main()