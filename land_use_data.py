# ============================================================
# land_use_data.py — Austin Crash Safety Prediction System
# Phase 1: Land Use & Built Environment Feature Extraction
#
# Features added to each crash record:
#   Zone_Category    — Residential, Commercial, Industrial,
#                      Mixed Use, Civic, or Unknown
#   Zone_Type        — raw Austin zoning code (SF-3, CS, GR etc)
#   Dist_To_School   — meters to nearest school
#   Near_School      — 1 if crash within 300m of a school
#   Dist_To_Bus_Stop — meters to nearest CapMetro bus stop
#   Near_Bus_Stop    — 1 if crash within 150m of a bus stop
#
# Data Sources — all verified and confirmed working:
#
#   Zoning (address-based, no coordinates):
#     https://data.austintexas.gov/Building-and-Development/
#             Zoning-By-Address/nbzi-qabm
#     Columns used: FULL_STREET_NAME, ZONING_ZTYPE,
#                   BASE_ZONE, BASE_ZONE_CATEGORY
#     Download: austin_zoning.csv (manual download required)
#
#   Schools:
#     https://data.austintexas.gov/Health-and-Community-Services/
#             City-of-Austin-Schools-with-Data/63ig-4knr
#     Download via Socrata API automatically
#
#   Bus Stops (CapMetro GTFS stops.txt renamed):
#     capmetro_stops (renamed from stops.txt in CapMetro GTFS zip)
#     Columns: stop_id, stop_name, stop_lat, stop_lon
#     Download manually from:
#     https://data.texas.gov/browse?Dataset-Category_Agency=
#     Capital+Metropolitan+Transportation+Authority
#
# IMPORTANT — manual downloads required before first run:
#   1. Download Zoning By Address CSV → save as austin_zoning.csv
#   2. Rename stops.txt from CapMetro GTFS → capmetro_stops
#   Both files go in your project folder (crash_ml_project/)
# ============================================================

import pandas as pd
import numpy as np
import os
from sklearn.neighbors import BallTree

# ── File paths ────────────────────────────────────────────────
# All files must be CSV format in your project folder
ZONING_FILE    = "austin_zoning.csv"      # Manual download required
BUS_STOPS_FILE = "capmetro_stops.csv"     # Renamed from stops.txt → .csv
SCHOOLS_FILE   = "austin_schools.csv"     # Your local schools file
                                          # Change this name to match
                                          # whatever you saved it as

# ── Proximity thresholds ──────────────────────────────────────
SCHOOL_PROXIMITY_M   = 300   # 300m ≈ 3 city blocks
BUS_STOP_PROXIMITY_M = 150   # 150m ≈ 1.5 blocks (walking distance)

# ── Earth radius for BallTree distance conversion ────────────
EARTH_RADIUS_M = 6_371_000


# ── Zoning category simplification ───────────────────────────
def simplify_zone(zone_code):
    """
    Map a raw Austin zoning code to a simplified category.

    Austin zoning code prefixes:
        SF / MF / MH / RR  = Residential
        CS / GR / GO / LR  = Commercial
        LI / MI / W/        = Industrial
        MU / CR / TOD / VMU = Mixed Use
        P / DR / AG         = Civic / Public
    """
    if not zone_code or pd.isna(zone_code):
        return "Unknown"
    code = str(zone_code).strip().upper()
    if code.startswith(("SF", "MF", "MH", "RR", "LA")):
        return "Residential"
    elif code.startswith(("CS", "GR", "GO", "LR", "CBD", "DMU")):
        return "Commercial"
    elif code.startswith(("LI", "MI", "W/")):
        return "Industrial"
    elif code.startswith(("MU", "CR", "TOD", "VMU", "NO")):
        return "Mixed Use"
    elif code.startswith(("P", "PDA", "DR", "AG")):
        return "Civic / Public"
    else:
        return "Other"


# ── Zoning Loader ─────────────────────────────────────────────
def _load_zoning():
    """
    Load Austin Zoning By Address dataset.

    This dataset uses street addresses not coordinates so we
    cannot build a BallTree from it directly. Instead we build
    a lookup dictionary keyed on street name so we can match
    crash report addresses to zone codes quickly.

    Confirmed columns:
        FULL_STREET_NAME  — e.g. "6021 CERVINUS RUN"
        ZONING_ZTYPE      — full zone code e.g. "SF-1-NP"
        BASE_ZONE         — base code e.g. "SF-1"
        BASE_ZONE_CATEGORY — e.g. "Single Family Large Lot"
    """
    if not os.path.exists(ZONING_FILE):
        print(f"  Zoning: {ZONING_FILE} not found in project folder.")
        print(f"  To get it:")
        print(f"  1. Go to: https://data.austintexas.gov/Building-and-Development/Zoning-By-Address/nbzi-qabm")
        print(f"  2. Click Export → Download as CSV")
        print(f"  3. Save as '{ZONING_FILE}' in your project folder")
        return pd.DataFrame(), {}

    print(f"  Zoning: Loading from {ZONING_FILE}...")
    df = pd.read_csv(ZONING_FILE, low_memory=False)

    # Confirm expected columns exist
    required = ["FULL_STREET_NAME", "ZONING_ZTYPE", "BASE_ZONE_CATEGORY"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        print(f"  Zoning: Missing expected columns: {missing}")
        print(f"  Available columns: {df.columns.tolist()}")
        return pd.DataFrame(), {}

    # Build street name lookup dict
    # Key = uppercase street name, Value = (zone_type, zone_category)
    lookup = {}
    for _, row in df.iterrows():
        street = str(row.get("FULL_STREET_NAME", "") or "").strip().upper()
        ztype  = str(row.get("ZONING_ZTYPE", "Unknown") or "Unknown")
        zcat   = str(row.get("BASE_ZONE_CATEGORY", "Unknown") or "Unknown")
        if street:
            lookup[street] = (ztype, simplify_zone(ztype))

    print(f"  Zoning: {len(df):,} records loaded, "
          f"{len(lookup):,} unique streets indexed")
    return df, lookup


# ── Schools Loader ────────────────────────────────────────────
def _load_schools():
    """
    Load Austin school locations from your local CSV file.
    No API download needed — uses the file you already have.

    If your file has a different name change SCHOOLS_FILE above.
    """
    if not os.path.exists(SCHOOLS_FILE):
        print(f"  Schools: '{SCHOOLS_FILE}' not found.")
        print(f"  Rename your schools CSV to '{SCHOOLS_FILE}' "
              f"and place it in your project folder.")
        return pd.DataFrame()

    print(f"  Schools: Loading from {SCHOOLS_FILE}...")
    try:
        df = pd.read_csv(SCHOOLS_FILE, low_memory=False)
    except Exception as e:
        print(f"  Schools: Failed to load — {e}")
        return pd.DataFrame()

    # Find lat/lon columns — handles different column name formats
    lat_col = lon_col = None
    for col in df.columns:
        cl = col.lower()
        if "lat" in cl and lat_col is None:
            lat_col = col
        if "lon" in cl and lon_col is None:
            lon_col = col

    if lat_col is None:
        print(f"  Schools: No lat/lon columns found.")
        print(f"  Columns in file: {df.columns.tolist()}")
        return pd.DataFrame()

    df = df.rename(columns={lat_col: "lat", lon_col: "lon"})
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    df = df[
        df["lat"].between(30.0, 30.7) &
        df["lon"].between(-98.1, -97.4)
    ]

    print(f"  Schools: {len(df)} locations loaded")
    return df[["lat", "lon"]]


# ── Bus Stops Loader ──────────────────────────────────────────
def _load_bus_stops():
    """
    Load CapMetro bus stop locations from GTFS stops file.

    Confirmed columns from stops.txt:
        stop_id, stop_name, stop_lat, stop_lon,
        at_street, on_street, heading, stop_code,
        stop_desc, location_type, parent_station,
        stop_position, stop_timezone, stop_url,
        wheelchair_boarding, zone_id

    The file was renamed from stops.txt to capmetro_stops.
    Place it in your project folder before running.
    """
    if not os.path.exists(BUS_STOPS_FILE):
        print(f"  Bus Stops: '{BUS_STOPS_FILE}' not found.")
        print(f"  To get it:")
        print(f"  1. Go to: https://data.texas.gov/browse?"
              f"Dataset-Category_Agency=Capital+Metropolitan"
              f"+Transportation+Authority")
        print(f"  2. Download the CapMetro GTFS zip file")
        print(f"  3. Unzip it and rename stops.txt → capmetro_stops")
        print(f"  4. Move capmetro_stops into your project folder")
        return pd.DataFrame()

    print(f"  Bus Stops: Loading from {BUS_STOPS_FILE}...")
    try:
        df = pd.read_csv(BUS_STOPS_FILE)
    except Exception as e:
        print(f"  Bus Stops: Failed to load — {e}")
        return pd.DataFrame()

    # Confirm expected columns
    if "stop_lat" not in df.columns or "stop_lon" not in df.columns:
        print(f"  Bus Stops: Missing stop_lat/stop_lon columns.")
        print(f"  Columns found: {df.columns.tolist()}")
        return pd.DataFrame()

    df = df.rename(columns={"stop_lat": "lat", "stop_lon": "lon"})
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])

    # Filter to Austin area only
    df = df[
        df["lat"].between(30.0, 30.7) &
        df["lon"].between(-98.1, -97.4)
    ]

    print(f"  Bus Stops: {len(df):,} stops loaded")
    return df[["lat", "lon"]]


# ── BallTree Builder ──────────────────────────────────────────
def _build_tree(df):
    """Build a BallTree spatial index for fast nearest-neighbor search."""
    if df.empty or len(df) < 1:
        return None
    coords = np.radians(df[["lat", "lon"]].values)
    return BallTree(coords, metric="haversine")


# ── Module-level Load ─────────────────────────────────────────
print("Loading land use data...")

_zoning_df, _zoning_lookup = _load_zoning()
_schools_df   = _load_schools()
_bus_stops_df = _load_bus_stops()

_schools_tree   = _build_tree(_schools_df)
_bus_stops_tree = _build_tree(_bus_stops_df)

print("Land use data ready.")


# ── Zone Lookup Helper ────────────────────────────────────────
def _lookup_zone(road_name, crash_address=None):
    """
    Look up zone type for a crash location using road name.
    Tries crash road name first then falls back to crash address.

    Returns (zone_type, zone_category) or ('Unknown', 'Unknown')
    """
    if not _zoning_lookup:
        return "Unknown", "Unknown"

    # Try road name first
    if road_name and pd.notna(road_name):
        key = str(road_name).strip().upper()
        if key in _zoning_lookup:
            return _zoning_lookup[key]
        # Try partial match — street name without block number
        parts = key.split()
        if len(parts) > 1:
            # Remove leading block number if present
            if parts[0].isdigit():
                partial = " ".join(parts[1:])
                for k, v in _zoning_lookup.items():
                    if partial in k:
                        return v

    # Try crash address fallback
    if crash_address and pd.notna(crash_address):
        key = str(crash_address).strip().upper()
        if key in _zoning_lookup:
            return _zoning_lookup[key]

    return "Unknown", "Unknown"


# ── Main Lookup Function ──────────────────────────────────────
def get_land_use(lat, lon, road_name=None, crash_address=None):
    """
    Return land use features for a crash location.

    Parameters:
        lat           — crash latitude
        lon           — crash longitude
        road_name     — road name from OSMnx (for zone lookup)
        crash_address — crash report address (fallback for zone)

    Returns tuple of 6 values:
        zone_category   — Residential / Commercial / Industrial /
                          Mixed Use / Civic / Unknown
        zone_type       — raw Austin zoning code (SF-3, CS etc)
        dist_school     — meters to nearest school (float or None)
        near_school     — 1 if within 300m, else 0
        dist_bus_stop   — meters to nearest bus stop (float or None)
        near_bus_stop   — 1 if within 150m, else 0
    """
    point = np.radians([[lat, lon]])

    # ── Zoning ────────────────────────────────────────────────
    zone_type, zone_category = _lookup_zone(road_name, crash_address)

    # ── Schools ───────────────────────────────────────────────
    dist_school = None
    near_school = 0
    if _schools_tree is not None:
        try:
            dist_rad, _ = _schools_tree.query(point, k=1)
            dist_school = round(dist_rad[0][0] * EARTH_RADIUS_M, 1)
            near_school = 1 if dist_school <= SCHOOL_PROXIMITY_M else 0
        except Exception as e:
            print(f"  School lookup failed ({lat}, {lon}): {e}")

    # ── Bus Stops ─────────────────────────────────────────────
    dist_bus_stop = None
    near_bus_stop = 0
    if _bus_stops_tree is not None:
        try:
            dist_rad, _  = _bus_stops_tree.query(point, k=1)
            dist_bus_stop = round(dist_rad[0][0] * EARTH_RADIUS_M, 1)
            near_bus_stop = (
                1 if dist_bus_stop <= BUS_STOP_PROXIMITY_M else 0
            )
        except Exception as e:
            print(f"  Bus stop lookup failed ({lat}, {lon}): {e}")

    return (
        zone_category,
        zone_type,
        dist_school,
        near_school,
        dist_bus_stop,
        near_bus_stop
    )


# ── Row-wise Wrapper ──────────────────────────────────────────
def process_land_use(row):
    """Row-wise wrapper for use with df.apply()"""
    (zone_cat, zone_typ,
     dist_sch, near_sch,
     dist_bus, near_bus) = get_land_use(
        row["latitude"],
        row["longitude"],
        road_name     = row.get("Road_Name"),
        crash_address = row.get("Address")
    )
    return pd.Series({
        "Zone_Category":   zone_cat,
        "Zone_Type":       zone_typ,
        "Dist_To_School":  dist_sch,
        "Near_School":     near_sch,
        "Dist_To_Bus_Stop": dist_bus,
        "Near_Bus_Stop":   near_bus
    })


# ── Verification ──────────────────────────────────────────────
def verify_data():
    """Print a summary of what was loaded successfully."""
    print("\n── Land Use Data Summary ──────────────────────────")
    print(f"  Zoning streets indexed : {len(_zoning_lookup):,}")
    print(f"  School locations       : {len(_schools_df):,}")
    print(f"  Bus stop locations     : {len(_bus_stops_df):,}")

    if _zoning_lookup:
        from collections import Counter
        cats = Counter(v[1] for v in _zoning_lookup.values())
        print("\n  Zone categories:")
        for cat, cnt in cats.most_common():
            print(f"    {cat:<22} {cnt:,}")

    missing = []
    if not _zoning_lookup:
        missing.append("austin_zoning.csv")
    if _bus_stops_df.empty:
        missing.append("capmetro_stops")
    if missing:
        print(f"\n  Missing files: {missing}")
        print("  See instructions at top of land_use_data.py")
    print("───────────────────────────────────────────────────\n")
