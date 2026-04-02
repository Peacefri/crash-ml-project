# ============================================================
# aadt_data.py — Austin Crash Safety Prediction System
# AADT (Annual Average Daily Traffic) Feature Extraction
# Normalizes crash rates by traffic exposure
#
# FIX: Distance-only matching replaced with road-name matching
# + road type fallback. Prevents assigning highway AADT to
# crashes that happened on completely different nearby roads.
# ============================================================

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.neighbors import BallTree

AADT_FILE  = "txdot_aadt.csv"
CACHE_FILE = "aadt_cache.pkl"

# Austin metro counties
AUSTIN_COUNTIES = [
    "Travis", "Williamson", "Hays", "Bastrop", "Caldwell"
]

# Road type average AADT estimates based on TxDOT/FHWA standards.
# Used as fallback when no matching station exists on the same road.
ROAD_TYPE_AADT = {
    "motorway":       150000,
    "motorway_link":   45000,
    "trunk":           80000,
    "trunk_link":      25000,
    "primary":         35000,
    "primary_link":    10000,
    "secondary":       15000,
    "secondary_link":   5000,
    "tertiary":         8000,
    "tertiary_link":    3000,
    "residential":      1500,
    "living_street":     500,
    "service":           500,
    "track":             200,
    "path":              100,
    "unclassified":     2000,
    "unknown":          5000
}


# ── Load and Filter AADT Data ────────────────────────────────
def load_aadt_data():
    """
    Load TxDOT AADT station data filtered to Austin metro area.
    Builds a BallTree spatial index for fast nearest-station lookup.
    Caches result to disk after first load.
    """
    if os.path.exists(CACHE_FILE):
        print("Loading AADT data from cache...")
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)

    print("Loading TxDOT AADT dataset...")
    df = pd.read_csv(AADT_FILE)

    aadt = df[
        (df["CNTY_NM"].isin(AUSTIN_COUNTIES)) &
        (df["ACTIVE"] == 1) &
        df["LATITUDE"].notna() &
        df["LONGITUDE"].notna() &
        df["AADT_RPT_QTY"].notna()
    ].copy().reset_index(drop=True)

    print(f"  Austin metro AADT stations loaded: {len(aadt)}")
    print(f"  Most recent report year: {aadt['AADT_RPT_YEAR'].max()}")
    print(f"  Roads covered: {aadt['ON_ROAD'].nunique()} unique roads")

    if len(aadt) == 0:
        raise ValueError(
            "No AADT stations found after filtering. "
            "Check AADT_FILE path and county names."
        )

    # Normalize road names to uppercase for consistent matching
    aadt["ON_ROAD_UPPER"] = aadt["ON_ROAD"].str.strip().str.upper()

    # Build BallTree spatial index using haversine distance
    coords_rad = np.radians(aadt[["LATITUDE", "LONGITUDE"]].values)
    tree = BallTree(coords_rad, metric="haversine")

    result = {"aadt": aadt, "tree": tree}

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(result, f)
    print("  AADT data cached to disk.")

    return result


# Load once when module is first imported
_data    = load_aadt_data()
_aadt_df = _data["aadt"]
_tree    = _data["tree"]


# ── Historical AADT Lookup ───────────────────────────────────
def get_historical_aadt(station_row, crash_year):
    """
    Return the AADT value closest to the crash year using
    historical columns (up to 19 years back from report year).
    """
    report_year = int(station_row["AADT_RPT_YEAR"])
    years_back  = report_year - int(crash_year)

    if years_back <= 0:
        return station_row["AADT_RPT_QTY"]

    if 1 <= years_back <= 19:
        hist_col = f"AADT_RPT_HIST_{str(years_back).zfill(2)}_QTY"
        val = station_row.get(hist_col, None)
        if pd.notna(val) and val > 0:
            return val

    # Outside range — use oldest available historical value
    for i in range(19, 0, -1):
        hist_col = f"AADT_RPT_HIST_{str(i).zfill(2)}_QTY"
        val = station_row.get(hist_col, None)
        if pd.notna(val) and val > 0:
            return val

    return station_row["AADT_RPT_QTY"]


# ── Road Name Matcher ────────────────────────────────────────
def roads_match(crash_road, station_road):
    """
    Check if a crash road name and station road name refer to
    the same road. Uses partial matching to handle abbreviations
    and formatting differences (e.g. 'LAMAR BLVD' vs 'N LAMAR').

    Returns True if roads are considered the same, False otherwise.
    """
    if not crash_road or not station_road:
        return False

    crash_road   = str(crash_road).strip().upper()
    station_road = str(station_road).strip().upper()

    if not crash_road or not station_road:
        return False

    # Direct match
    if crash_road == station_road:
        return True

    # Partial match — one name contains the other
    if crash_road in station_road or station_road in crash_road:
        return True

    # Check if key words match (ignores directional prefixes N/S/E/W)
    crash_words   = set(crash_road.split())
    station_words = set(station_road.split())
    prefixes      = {"N", "S", "E", "W", "NB", "SB", "EB", "WB",
                     "NORTH", "SOUTH", "EAST", "WEST"}
    crash_core    = crash_words - prefixes
    station_core  = station_words - prefixes

    if crash_core and station_core and crash_core & station_core:
        return True

    return False


# ── Main AADT Lookup Function ────────────────────────────────
def get_aadt(lat, lon, crash_year, road_name=None,
             highway_type=None, max_distance_km=1.0):
    """
    Find AADT for a crash using a two-step approach:

    Step 1 — Station matching with road name verification:
        Search up to 5 nearest stations within max_distance_km.
        Only accept a station if its road name matches the crash
        road name. This prevents assigning I-35 AADT to a crash
        that happened on a nearby residential street.

    Step 2 — Road type fallback:
        If no same-road station exists, estimate AADT from the
        road functional class average (TxDOT/FHWA standard approach).

    Parameters:
        lat             - crash latitude
        lon             - crash longitude
        crash_year      - year crash occurred (for historical AADT)
        road_name       - name of road where crash happened (from OSMnx)
        highway_type    - OSMnx highway tag (e.g. 'primary', 'residential')
        max_distance_km - maximum search radius in km (default 1.0km)

    Returns a tuple of 5 values:
        aadt_value    - vehicles per day (float or None)
        station_road  - matched station road name (or None)
        distance_km   - distance to matched station (or None)
        aadt_source   - 'station', 'road_type_estimate', or 'no_match'
        highway_used  - highway type used for estimate (or None)
    """

    # ── Step 1: Station matching with road name verification ──
    try:
        point_rad = np.radians([[lat, lon]])

        # Get 5 nearest stations to find best road name match
        k         = min(5, len(_aadt_df))
        dist_rad, idx = _tree.query(point_rad, k=k)

        for i in range(k):
            dist_km = dist_rad[0][i] * 6371  # radians to km

            # Stop if beyond search radius
            if dist_km > max_distance_km:
                break

            station      = _aadt_df.iloc[idx[0][i]]
            station_road = station.get("ON_ROAD_UPPER", "")

            # Only accept if road names match
            if roads_match(road_name, station_road):
                aadt_value = get_historical_aadt(station, crash_year)
                return (
                    float(aadt_value),
                    station_road,
                    round(dist_km, 3),
                    "station",
                    None
                )

    except Exception as e:
        print(f"  AADT station lookup failed for ({lat}, {lon}): {e}")

    # ── Step 2: Road type fallback ────────────────────────────
    if highway_type and highway_type in ROAD_TYPE_AADT:
        estimated = ROAD_TYPE_AADT[highway_type]
        return (
            float(estimated),
            None,
            None,
            "road_type_estimate",
            highway_type
        )

    # No station match and no road type available
    return None, None, None, "no_match", None


# ── Crash Rate Calculator ────────────────────────────────────
def calculate_crash_rate(crash_count, aadt, segment_length_km=0.1):
    """
    Calculate crash rate per million vehicle miles traveled (MVMT).
    Standard metric used by FHWA and TxDOT.

    Formula:
        Crash Rate = (Crashes x 1,000,000) / (AADT x 365 x Length_miles)
    """
    if aadt is None or aadt <= 0:
        return None

    length_miles = segment_length_km * 0.621371
    annual_vmt   = aadt * 365 * length_miles

    if annual_vmt <= 0:
        return None

    return round((crash_count * 1_000_000) / annual_vmt, 4)


# ── Row-wise Wrapper ─────────────────────────────────────────
def process_aadt(row):
    """Row-wise wrapper for use with df.apply()"""
    lat       = row.get("latitude")
    lon       = row.get("longitude")
    timestamp = row.get("Crash timestamp (US/Central)")

    crash_year = None
    if pd.notna(timestamp):
        parsed = pd.to_datetime(timestamp, errors="coerce")
        if pd.notna(parsed):
            crash_year = parsed.year

    if pd.isna(lat) or pd.isna(lon) or crash_year is None:
        return pd.Series({
            "AADT":              None,
            "AADT_Station_Road": None,
            "AADT_Distance_km":  None,
            "AADT_Source":       "no_match",
            "AADT_Year_Used":    None
        })

    aadt_val, station_road, dist_km, source, _ = get_aadt(
        lat, lon, crash_year,
        road_name    = row.get("Road_Name"),
        highway_type = row.get("Highway_Type")
    )

    return pd.Series({
        "AADT":              aadt_val,
        "AADT_Station_Road": station_road,
        "AADT_Distance_km":  dist_km,
        "AADT_Source":       source,
        "AADT_Year_Used":    crash_year
    })