# ============================================================
# main.py — Austin Crash Safety Prediction System
# Phase 1: Data Collection & Enrichment
#
# Features collected:
#   Road    : Highway_Type, Road_Type_Label, Road_Name,
#             Num_Lanes, Speed_Limit, Road_Risk_Level,
#             Is_Intersection, Intersection_Degree,
#             Road_Curvature, Street_Lit
#   AADT    : AADT, AADT_Station_Road,
#             AADT_Distance_km, AADT_Source
#   Weather : Temperature, Precipitation, Windspeed,
#             Visibility, Weather_Code,
#             Weather_Condition, is_wet
#   Time    : Crash Date, Hour, Day, Month,
#             Year, Is_Weekend, Is_Dark
#   Lighting: Street_Lit, Dark_Unlit, Dark_Lit
#   Land Use: Zone_Category, Zone_Type,
#             Dist_To_School, Near_School,
#             Dist_To_Bus_Stop, Near_Bus_Stop
#   Severity: Severity_Label, Is_Severe
# ============================================================

import pandas as pd
import time
import os

from road_data import get_road_type
from weather_data import get_weather, decode_weathercode
from visuals_data import create_visualizations
from aadt_data import get_aadt
from land_use_data import get_land_use


# ── File paths ───────────────────────────────────────────────
PROJECT_DIR     = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE      = os.environ.get("CRASH_INPUT_FILE", os.path.join(PROJECT_DIR, "crashes_raw.csv"))
OUTPUT_FILE     = os.environ.get("CRASH_OUTPUT_FILE", os.path.join(PROJECT_DIR, "crashes_final_enriched.csv"))
CHECKPOINT_FILE = os.environ.get("CRASH_CHECKPOINT_FILE", os.path.join(PROJECT_DIR, "crashes_checkpoint.csv"))


# ── 1. Load Data ─────────────────────────────────────────────
def load_data():
    """
    Load crash CSV using the file's own header row.

    FIX: This previously passed a hardcoded 47-name `names=` list. Whenever
    the input had more columns than that, pandas silently absorbed the extras
    into the index and shifted every remaining column — 'latitude' ended up
    holding injury counts and the timestamp column a dollar amount, so every
    downstream lookup failed. Trust the file's header; validate it instead.

    Sample size is controlled by fetch_data.py (--crashes N), not here,
    so the whole input file is always processed.
    """
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df.reset_index(drop=True)

    # Fail loudly if the file is not shaped the way the pipeline expects,
    # rather than silently producing NaN for every enriched feature.
    required = [
        "latitude", "longitude", "crash_sev_id",
        "Crash timestamp (US/Central)"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input file '{INPUT_FILE}' is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    print(f"Dataset loaded — {len(df)} rows, {len(df.columns)} columns")
    return df


# ── 2. Time Features ─────────────────────────────────────────
def create_time_features(df):
    # FIX: the timestamp column appears in two formats depending on which
    # stage produced the file — ISO ("2014-01-08 13:35:00") and the TxDOT
    # export style ("2014 Jan 08 07:35:00 PM"). Hardcoding the second one
    # turned every row into NaT, which cascaded into NaN crash_year/hour and
    # blew up the AADT and weather lookups. Try the explicit format, then
    # fall back to inferred parsing for whatever it could not handle.
    raw_ts = df["Crash timestamp (US/Central)"]
    parsed = pd.to_datetime(raw_ts, format="%Y %b %d %I:%M:%S %p", errors="coerce")

    still_bad = parsed.isna() & raw_ts.notna()
    if still_bad.any():
        parsed.loc[still_bad] = pd.to_datetime(raw_ts[still_bad], errors="coerce")

    df["Crash timestamp (US/Central)"] = parsed

    bad_dates = df["Crash timestamp (US/Central)"].isna().sum()
    if bad_dates > 0:
        print(f"  Warning: {bad_dates} timestamps could not be parsed")

    df["Crash Date"]  = df["Crash timestamp (US/Central)"].dt.date
    df["Crash Hour"]  = df["Crash timestamp (US/Central)"].dt.hour
    df["Crash Day"]   = df["Crash timestamp (US/Central)"].dt.day_name()
    df["Crash Month"] = df["Crash timestamp (US/Central)"].dt.month
    df["Crash Year"]  = df["Crash timestamp (US/Central)"].dt.year
    df["Is_Weekend"]  = df["Crash Day"].isin(["Saturday", "Sunday"])
    df["Is_Dark"]     = df["Crash Hour"].apply(
        lambda h: 1 if pd.notna(h) and (h >= 20 or h <= 6) else 0
    )

    print("  Time features: Date, Hour, Day, Month, Year, Is_Weekend, Is_Dark")
    return df


# ── 3. Severity Label ────────────────────────────────────────
def create_severity_label(df):
    severity_labels = {
        0: "Unknown",
        1: "Incapacitating Injury",
        2: "Non-Incapacitating Injury",
        3: "Possible Injury",
        4: "Killed",
        5: "Not Injured"
    }
    df["Severity_Label"] = df["crash_sev_id"].map(severity_labels).fillna("Unknown")
    df["Is_Severe"]      = df["crash_sev_id"].isin([0, 1, 4]).astype(int)
    print("  Severity labels and Is_Severe target column created")
    return df


# ── 4. Enrich Data ───────────────────────────────────────────
def enrich_data(df):
    """
    Loop through every crash row calling road, AADT, weather,
    and land use APIs to attach enriched features to each record.
    """

    # ── Checkpoint Recovery ───────────────────────────────────
    # FIX: Pre-populate lists from checkpoint so lengths always
    # match df when attaching columns at the end of the loop.
    if os.path.exists(CHECKPOINT_FILE):
        print(f"\n  Checkpoint found — resuming from saved progress...")
        checkpoint_df   = pd.read_csv(CHECKPOINT_FILE)
        processed_count = len(checkpoint_df)
        print(f"  Resuming from row {processed_count}\n")

        highways             = checkpoint_df.get("Highway_Type",        pd.Series()).tolist()
        highway_labels       = checkpoint_df.get("Road_Type_Label",     pd.Series()).tolist()
        road_names           = checkpoint_df.get("Road_Name",           pd.Series()).tolist()
        lanes_list           = checkpoint_df.get("Num_Lanes",           pd.Series()).tolist()
        speed_limits         = checkpoint_df.get("Speed_Limit",         pd.Series()).tolist()
        road_risks           = checkpoint_df.get("Road_Risk_Level",     pd.Series()).tolist()
        is_intersections     = checkpoint_df.get("Is_Intersection",     pd.Series()).tolist()
        intersection_degrees = checkpoint_df.get("Intersection_Degree", pd.Series()).tolist()
        curvatures           = checkpoint_df.get("Road_Curvature",      pd.Series()).tolist()
        lit_values           = checkpoint_df.get("Street_Lit",          pd.Series()).tolist()
        aadt_values          = checkpoint_df.get("AADT",                pd.Series()).tolist()
        aadt_roads           = checkpoint_df.get("AADT_Station_Road",   pd.Series()).tolist()
        aadt_distances       = checkpoint_df.get("AADT_Distance_km",    pd.Series()).tolist()
        aadt_sources         = checkpoint_df.get("AADT_Source",         pd.Series()).tolist()
        temps                = checkpoint_df.get("Temperature",         pd.Series()).tolist()
        precips              = checkpoint_df.get("Precipitation",       pd.Series()).tolist()
        windspeeds           = checkpoint_df.get("Windspeed",           pd.Series()).tolist()
        visibilities         = checkpoint_df.get("Visibility",          pd.Series()).tolist()
        weathercodes         = checkpoint_df.get("Weather_Code",        pd.Series()).tolist()
        is_wet_list          = checkpoint_df.get("is_wet",              pd.Series()).tolist()
        zone_categories      = checkpoint_df.get("Zone_Category",       pd.Series()).tolist()
        zone_types           = checkpoint_df.get("Zone_Type",           pd.Series()).tolist()
        dist_schools         = checkpoint_df.get("Dist_To_School",      pd.Series()).tolist()
        near_schools         = checkpoint_df.get("Near_School",         pd.Series()).tolist()
        dist_bus_stops       = checkpoint_df.get("Dist_To_Bus_Stop",    pd.Series()).tolist()
        near_bus_stops       = checkpoint_df.get("Near_Bus_Stop",       pd.Series()).tolist()
    else:
        processed_count      = 0
        highways             = []
        highway_labels       = []
        road_names           = []
        lanes_list           = []
        speed_limits         = []
        road_risks           = []
        is_intersections     = []
        intersection_degrees = []
        curvatures           = []
        lit_values           = []
        aadt_values          = []
        aadt_roads           = []
        aadt_distances       = []
        aadt_sources         = []
        temps                = []
        precips              = []
        windspeeds           = []
        visibilities         = []
        weathercodes         = []
        is_wet_list          = []
        zone_categories      = []
        zone_types           = []
        dist_schools         = []
        near_schools         = []
        dist_bus_stops       = []
        near_bus_stops       = []

    total = len(df)

    for loop_idx, (df_idx, row) in enumerate(df.iterrows()):

        if loop_idx < processed_count:
            continue

        lat = row.get("latitude")
        lon = row.get("longitude")

        # ── Missing coordinates ───────────────────────────────
        if pd.isna(lat) or pd.isna(lon):
            print(f"  Row {loop_idx}: Missing coordinates — skipping")
            highways.append(None)
            highway_labels.append(None)
            road_names.append(None)
            lanes_list.append(None)
            speed_limits.append(None)
            road_risks.append(None)
            is_intersections.append(None)
            intersection_degrees.append(None)
            curvatures.append(None)
            lit_values.append(None)
            aadt_values.append(None)
            aadt_roads.append(None)
            aadt_distances.append(None)
            aadt_sources.append("no_match")
            temps.append(None)
            precips.append(None)
            windspeeds.append(None)
            visibilities.append(None)
            weathercodes.append(None)
            is_wet_list.append(None)
            zone_categories.append("Unknown")
            zone_types.append("Unknown")
            dist_schools.append(None)
            near_schools.append(0)
            dist_bus_stops.append(None)
            near_bus_stops.append(0)
            continue

        # ── Road Data ─────────────────────────────────────────
        try:
            (highway, highway_label, lanes, road_risk, speed,
             is_intersection, intersection_degree,
             curvature, road_name, lit) = get_road_type(lat, lon)
        except Exception as e:
            print(f"  Row {loop_idx}: Road data failed — {e}")
            (highway, highway_label, lanes, road_risk, speed,
             is_intersection, intersection_degree,
             curvature, road_name, lit) = (
                None, None, None, None, None,
                None, None, None, None, None
            )

        highways.append(highway)
        highway_labels.append(highway_label)

        # If OSMnx road_name is None fall back to crash report address
        if road_name is None:
            block  = str(row.get("rpt_block_num",   "") or "").strip()
            street = str(row.get("rpt_street_name", "") or "").strip()
            suffix = str(row.get("rpt_street_sfx",  "") or "").strip()
            parts  = [p for p in [block, street, suffix] if p]
            road_name = " ".join(parts) if parts else None

        road_names.append(road_name)
        lanes_list.append(lanes)
        speed_limits.append(speed)
        road_risks.append(road_risk)
        is_intersections.append(is_intersection)
        intersection_degrees.append(intersection_degree)
        curvatures.append(curvature)
        lit_values.append(lit)

        # ── AADT Data ─────────────────────────────────────────
        try:
            crash_year = row.get("Crash Year")
            if pd.isna(crash_year):
                crash_year = pd.to_datetime(
                    row["Crash timestamp (US/Central)"], errors="coerce"
                ).year

            # FIX: guard the int() cast. An unparseable timestamp left this as
            # NaN and raised "cannot convert float NaN to integer" for the row.
            if pd.isna(crash_year):
                raise ValueError("no usable crash year (unparseable timestamp)")

            aadt_val, aadt_road, aadt_dist, aadt_source, _ = get_aadt(
                lat, lon, int(crash_year),
                road_name    = road_name,
                highway_type = highway
            )
        except Exception as e:
            print(f"  Row {loop_idx}: AADT lookup failed — {e}")
            aadt_val, aadt_road, aadt_dist, aadt_source = (
                None, None, None, "no_match"
            )

        aadt_values.append(aadt_val)
        aadt_roads.append(aadt_road)
        aadt_distances.append(aadt_dist)
        aadt_sources.append(aadt_source)

        # ── Weather Data ──────────────────────────────────────
        try:
            # FIX: get_weather() does int(hour) internally, so a NaN hour or a
            # NaT date raised "cannot convert float NaN to integer". Check here
            # so the row degrades to null weather instead of erroring.
            crash_date = row.get("Crash Date")
            crash_hour = row.get("Crash Hour")
            if pd.isna(crash_date) or pd.isna(crash_hour):
                raise ValueError("no usable crash date/hour (unparseable timestamp)")

            temp, precip, windspeed, visibility, weathercode = get_weather(
                lat, lon, str(crash_date), int(crash_hour)
            )
        except Exception as e:
            print(f"  Row {loop_idx}: Weather data failed — {e}")
            temp, precip, windspeed, visibility, weathercode = (
                None, None, None, None, None
            )

        temps.append(temp)
        precips.append(precip)
        windspeeds.append(windspeed)
        visibilities.append(visibility)
        weathercodes.append(weathercode)
        is_wet_list.append(precip > 0 if precip is not None else None)

        # ── Land Use Data ─────────────────────────────────────
        try:
            (zone_cat, zone_typ,
             dist_sch, near_sch,
             dist_bus, near_bus) = get_land_use(
                lat, lon,
                road_name     = road_name,
                crash_address = row.get("Address")
            )
        except Exception as e:
            print(f"  Row {loop_idx}: Land use lookup failed — {e}")
            zone_cat = "Unknown"
            zone_typ = "Unknown"
            dist_sch = None
            near_sch = 0
            dist_bus = None
            near_bus = 0

        zone_categories.append(zone_cat)
        zone_types.append(zone_typ)
        dist_schools.append(dist_sch)
        near_schools.append(near_sch)
        dist_bus_stops.append(dist_bus)
        near_bus_stops.append(near_bus)

        if (loop_idx + 1) % 500 == 0 or (loop_idx + 1) == total:
            print(f"  Progress: {loop_idx + 1}/{total} rows processed...")

        if (loop_idx + 1) % 500 == 0:
            _save_checkpoint(
                df, loop_idx,
                highways, highway_labels, road_names,
                lanes_list, speed_limits, road_risks,
                is_intersections, intersection_degrees, curvatures,
                lit_values,
                aadt_values, aadt_roads, aadt_distances, aadt_sources,
                temps, precips, windspeeds, visibilities,
                weathercodes, is_wet_list,
                zone_categories, zone_types,
                dist_schools, near_schools,
                dist_bus_stops, near_bus_stops
            )

        if loop_idx % 10 == 0:
            time.sleep(0.5)

    # ── Attach enriched columns ───────────────────────────────

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
    df["Street_Lit"]           = lit_values

    # AADT columns
    df["AADT"]                 = aadt_values
    df["AADT_Station_Road"]    = aadt_roads
    df["AADT_Distance_km"]     = aadt_distances
    df["AADT_Source"]          = aadt_sources

    # Weather columns
    df["Temperature"]          = temps
    df["Precipitation"]        = precips
    df["Windspeed"]            = windspeeds
    df["Visibility"]           = visibilities
    df["Weather_Code"]         = weathercodes
    df["Weather_Condition"]    = df["Weather_Code"].apply(decode_weathercode)
    df["is_wet"]               = is_wet_list

    # Lighting interaction features
    df["Dark_Unlit"] = (
        (df["Is_Dark"] == 1) &
        (df["Street_Lit"] == "no")
    ).astype(int)

    df["Dark_Lit"] = (
        (df["Is_Dark"] == 1) &
        (df["Street_Lit"] == "yes")
    ).astype(int)

    # Land use columns
    df["Zone_Category"]    = zone_categories
    df["Zone_Type"]        = zone_types
    df["Dist_To_School"]   = dist_schools
    df["Near_School"]      = near_schools
    df["Dist_To_Bus_Stop"] = dist_bus_stops
    df["Near_Bus_Stop"]    = near_bus_stops

    # Print Street_Lit coverage summary
    lit_counts = df["Street_Lit"].value_counts(dropna=False)
    print(f"\n  Street_Lit coverage:")
    print(f"    yes (confirmed lit)  : {lit_counts.get('yes', 0)}")
    print(f"    no  (confirmed unlit): {lit_counts.get('no', 0)}")
    print(f"    None (not tagged)    : {df['Street_Lit'].isna().sum()}")
    print(f"    Dark_Unlit crashes   : {df['Dark_Unlit'].sum()}")
    print(f"    Dark_Lit crashes     : {df['Dark_Lit'].sum()}")

    # Print land use coverage summary
    print(f"\n  Land use coverage:")
    print(f"    Zone categories: "
          f"{df['Zone_Category'].value_counts().to_dict()}")
    print(f"    Near school    : {df['Near_School'].sum()} crashes")
    print(f"    Near bus stop  : {df['Near_Bus_Stop'].sum()} crashes")

    return df


# ── Checkpoint Helper ────────────────────────────────────────
def _save_checkpoint(df, loop_idx,
                     highways, highway_labels, road_names,
                     lanes_list, speed_limits, road_risks,
                     is_intersections, intersection_degrees, curvatures,
                     lit_values,
                     aadt_values, aadt_roads, aadt_distances, aadt_sources,
                     temps, precips, windspeeds, visibilities,
                     weathercodes, is_wet_list,
                     zone_categories, zone_types,
                     dist_schools, near_schools,
                     dist_bus_stops, near_bus_stops):
    n       = loop_idx + 1
    partial = df.iloc[:n].copy()

    def pad(lst, default=None):
        return (lst + [default] * n)[:n]

    partial["Highway_Type"]        = pad(highways)
    partial["Road_Type_Label"]     = pad(highway_labels)
    partial["Road_Name"]           = pad(road_names)
    partial["Num_Lanes"]           = pad(lanes_list)
    partial["Speed_Limit"]         = pad(speed_limits)
    partial["Road_Risk_Level"]     = pad(road_risks)
    partial["Is_Intersection"]     = pad(is_intersections)
    partial["Intersection_Degree"] = pad(intersection_degrees)
    partial["Road_Curvature"]      = pad(curvatures)
    partial["Street_Lit"]          = pad(lit_values)
    partial["AADT"]                = pad(aadt_values)
    partial["AADT_Station_Road"]   = pad(aadt_roads)
    partial["AADT_Distance_km"]    = pad(aadt_distances)
    partial["AADT_Source"]         = pad(aadt_sources)
    partial["Temperature"]         = pad(temps)
    partial["Precipitation"]       = pad(precips)
    partial["Windspeed"]           = pad(windspeeds)
    partial["Visibility"]          = pad(visibilities)
    partial["Weather_Code"]        = pad(weathercodes)
    partial["is_wet"]              = pad(is_wet_list)
    partial["Zone_Category"]       = pad(zone_categories, "Unknown")
    partial["Zone_Type"]           = pad(zone_types,      "Unknown")
    partial["Dist_To_School"]      = pad(dist_schools)
    partial["Near_School"]         = pad(near_schools, 0)
    partial["Dist_To_Bus_Stop"]    = pad(dist_bus_stops)
    partial["Near_Bus_Stop"]       = pad(near_bus_stops, 0)

    partial.to_csv(CHECKPOINT_FILE, index=False)
    print(f"  Checkpoint saved at row {n}")


# ── 5. Main ──────────────────────────────────────────────────
def main():

    print("=" * 55)
    print("  Austin Crash Safety System — Phase 1 Enrichment")
    print("=" * 55)
    print(f"  Input file  : {os.path.abspath(INPUT_FILE)}")
    print(f"  Output file : {os.path.abspath(OUTPUT_FILE)}")

    print("\n[Step 1] Loading data...")
    df = load_data()

    print("\n[Step 2] Creating time features...")
    df = create_time_features(df)

    print("\n[Step 3] Creating severity labels...")
    df = create_severity_label(df)

    print("\n[Step 4] Enriching data (road + AADT + weather + land use)...")
    print("  Progress updates every 500 rows\n")
    df = enrich_data(df)

    print(f"\n[Step 5] Saving enriched dataset to '{OUTPUT_FILE}'...")
    output_dir = os.path.dirname(OUTPUT_FILE) or PROJECT_DIR
    os.makedirs(output_dir, exist_ok=True)
    temporary_output = f"{OUTPUT_FILE}.tmp"
    df.to_csv(temporary_output, index=False)
    os.replace(temporary_output, OUTPUT_FILE)
    with open(OUTPUT_FILE, "r", encoding="utf-8") as saved_file:
        saved_rows = sum(1 for _ in saved_file) - 1
    if saved_rows != len(df):
        raise RuntimeError(
            f"Output verification failed: expected {len(df)} rows, found {saved_rows}"
        )
    print(f"  Saved and verified — {saved_rows} rows, {len(df.columns)} columns")

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("  Checkpoint file removed")

    print("\n[Step 6] Generating PNG charts...")
    create_visualizations(df)

    print("\n" + "=" * 55)
    print("  Phase 1 Complete!")
    print(f"  Output file  : {OUTPUT_FILE}")
    print(f"  Total rows   : {len(df)}")
    print(f"  Total columns: {len(df.columns)}")
    print()
    print("  Road    : Highway_Type, Road_Type_Label, Road_Name,")
    print("            Num_Lanes, Speed_Limit, Road_Risk_Level,")
    print("            Is_Intersection, Intersection_Degree,")
    print("            Road_Curvature, Street_Lit")
    print("  AADT    : AADT, AADT_Station_Road,")
    print("            AADT_Distance_km, AADT_Source")
    print("  Weather : Temperature, Precipitation, Windspeed,")
    print("            Visibility, Weather_Code,")
    print("            Weather_Condition, is_wet")
    print("  Time    : Crash Date, Hour, Day, Month,")
    print("            Year, Is_Weekend, Is_Dark")
    print("  Lighting: Street_Lit, Dark_Unlit, Dark_Lit")
    print("  Land Use: Zone_Category, Zone_Type,")
    print("            Dist_To_School, Near_School,")
    print("            Dist_To_Bus_Stop, Near_Bus_Stop")
    print("  Severity: Severity_Label, Is_Severe")
    print()
    print("  PNG charts : see .png files in this folder")
    print("  HTML maps  : python crash_frequency_map.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
    