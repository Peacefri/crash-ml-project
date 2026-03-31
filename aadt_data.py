# ============================================================
# aadt_data.py — Austin Crash Safety Prediction System
# AADT (Annual Average Daily Traffic) Feature Extraction
# Normalizes crash rates by traffic exposure
# ============================================================

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.neighbors import BallTree

AADT_FILE  = "txdot_aadt.csv"
CACHE_FILE = "aadt_cache.pkl"

# Travis County and surrounding counties for Austin metro
AUSTIN_COUNTIES = [
    "Travis", "Williamson", "Hays", "Bastrop", "Caldwell"
]


# ── Load and Filter AADT Data ────────────────────────────────
def load_aadt_data():
    """
    Load TxDOT AADT station data and filter to Austin metro area.
    Builds a spatial index for fast nearest-station lookup.
    """
    # Use cached version if available
    if os.path.exists(CACHE_FILE):
        print("Loading AADT data from cache...")
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)

    print("Loading TxDOT AADT dataset...")
    df = pd.read_csv(AADT_FILE)

    # Filter to Austin metro counties AND active stations with valid coordinates
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
        raise ValueError("No AADT stations found after filtering. Check AADT_FILE and county names.")

    # Build spatial index (BallTree) for fast nearest-neighbor lookup
    # BallTree uses haversine distance — works correctly with lat/lon
    coords_rad = np.radians(aadt[["LATITUDE", "LONGITUDE"]].values)
    tree = BallTree(coords_rad, metric="haversine")

    result = {"aadt": aadt, "tree": tree}

    # Cache to disk so we don't rebuild every run
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
    Given a station row and a crash year, return the AADT
    value closest to that crash year using historical columns.

    The historical columns go back from AADT_RPT_YEAR:
        AADT_RPT_QTY      = most recent year (AADT_RPT_YEAR)
        AADT_RPT_HIST_01  = one year before
        AADT_RPT_HIST_02  = two years before
        ...
        AADT_RPT_HIST_19  = 19 years before
    """
    report_year = int(station_row["AADT_RPT_YEAR"])
    years_back  = report_year - int(crash_year)

    # Crash year is same as or more recent than report year
    if years_back <= 0:
        return station_row["AADT_RPT_QTY"]

    # Crash year is within historical range (up to 19 years back)
    if 1 <= years_back <= 19:
        hist_col = f"AADT_RPT_HIST_{str(years_back).zfill(2)}_QTY"
        val = station_row.get(hist_col, None)
        if pd.notna(val) and val > 0:
            return val

    # Outside historical range — use oldest available value
    for i in range(19, 0, -1):
        hist_col = f"AADT_RPT_HIST_{str(i).zfill(2)}_QTY"
        val = station_row.get(hist_col, None)
        if pd.notna(val) and val > 0:
            return val

    # Last resort — return most recent value
    return station_row["AADT_RPT_QTY"]


# ── Main AADT Lookup Function ────────────────────────────────
def get_aadt(lat, lon, crash_year, max_distance_km=2.0):
    """
    Given a crash location and year, find the nearest AADT
    traffic count station and return its traffic volume.

    Parameters:
        lat             - crash latitude
        lon             - crash longitude
        crash_year      - year the crash occurred (for historical AADT)
        max_distance_km - maximum search radius in km (default 2km)

    Returns a tuple of 4 values:
        aadt_value      - vehicles per day at crash year (or None)
        station_road    - name of road the station is on
        distance_km     - distance from crash to nearest station in km
        crash_rate_raw  - placeholder (calculated separately)
    """
    try:
        # Convert crash coordinates to radians for BallTree
        point_rad = np.radians([[lat, lon]])

        # Find nearest station
        dist_rad, idx = _tree.query(point_rad, k=1)

        # Convert radians to km (Earth radius = 6371 km)
        dist_km = dist_rad[0][0] * 6371

        # Nearest station is too far away
        if dist_km > max_distance_km:
            return None, None, round(dist_km, 3), None

        # Get the matching station row
        station = _aadt_df.iloc[idx[0][0]]

        # Get the AADT value closest to the crash year
        aadt_value   = get_historical_aadt(station, crash_year)
        station_road = station.get("ON_ROAD", None)

        return float(aadt_value), station_road, round(dist_km, 3), None

    except Exception as e:
        print(f"  AADT lookup failed for ({lat}, {lon}): {e}")
        return None, None, None, None


# ── Crash Rate Calculator ────────────────────────────────────
def calculate_crash_rate(crash_count, aadt, segment_length_km=0.1):
    """
    Calculate the crash rate per million vehicle miles traveled (MVMT).
    This is the standard metric used by FHWA and TxDOT.

    Formula:
        Crash Rate = (Crashes x 1,000,000) / (AADT x 365 x Length_miles)

    Parameters:
        crash_count       - number of crashes at this location
        aadt              - annual average daily traffic
        segment_length_km - road segment length in km (default 100m = 0.1km)

    Returns crash rate per million vehicle miles.
    """
    if aadt is None or aadt <= 0:
        return None

    # Convert km to miles
    length_miles = segment_length_km * 0.621371

    # Annual vehicle miles traveled on this segment
    annual_vmt = aadt * 365 * length_miles

    if annual_vmt <= 0:
        return None

    crash_rate = (crash_count * 1_000_000) / annual_vmt
    return round(crash_rate, 4)


# ── Row-wise Wrapper ─────────────────────────────────────────
def process_aadt(row):
    """
    Row-wise wrapper for use with df.apply().
    Returns a pandas Series with all AADT features.
    """
    lat        = row.get("latitude")
    lon        = row.get("longitude")
    timestamp  = row.get("Crash timestamp (US/Central)")

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
            "AADT_Year_Used":    None
        })

    aadt_val, station_road, dist_km, _ = get_aadt(lat, lon, crash_year)

    return pd.Series({
        "AADT":              aadt_val,
        "AADT_Station_Road": station_road,
        "AADT_Distance_km":  dist_km,
        "AADT_Year_Used":    crash_year
    })