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
#     NCES Public School Locations - Current, filtered to the five
#     Austin metro counties. Columns used: LAT, LON.
#     https://services1.arcgis.com/Ua5sjt3LWTPigjyD/arcgis/rest/
#             services/Public_School_Locations_Current/FeatureServer/0
#     NOTE: the City of Austin "Schools with Data" set (63ig-4knr)
#     was the original source here, but it publishes no coordinate
#     columns, so it cannot drive a distance calculation. NCES
#     covers public and charter schools; private schools are absent.
#
#   Bus Stops (CapMetro GTFS stops.txt):
#     Columns: stop_id, stop_name, stop_lat, stop_lon
#     https://data.texas.gov/dataset/CapMetro-GTFS/r4v4-vz24
#
# All three files are downloaded by fetch_data.py — run it once
# before main.py. They are gitignored, so a fresh clone has none
# of them:
#     .venv/Scripts/python.exe fetch_data.py
#
# FIXES:
#   - Partial zoning match now uses a pre-built suffix index
#     instead of looping through entire dictionary on each call
#     (prevents severe slowdown on 90k row full run)
#   - verify_data() now called automatically at module load
# ============================================================

import pandas as pd
import numpy as np
import os
from sklearn.neighbors import BallTree

# ── File paths ────────────────────────────────────────────────
PROJECT_DIR    = os.path.dirname(os.path.abspath(__file__))
ZONING_FILE    = os.path.join(PROJECT_DIR, "austin_zoning.csv")
BUS_STOPS_FILE = os.path.join(PROJECT_DIR, "capmetro_stops.csv")
SCHOOLS_FILE   = os.path.join(PROJECT_DIR, "austin_schools.csv")

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
        NO                  = Neighborhood Office (mapped to Mixed Use
                              as it serves as a transition zone between
                              residential and commercial areas)
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

    Builds two lookup dictionaries for fast zone matching:
      1. _zoning_lookup  — keyed on full street name (exact match)
      2. _zoning_suffix  — keyed on street name without block number
                           (partial match, pre-built to avoid O(n)
                           scan on every crash row lookup)

    Confirmed columns:
        FULL_STREET_NAME  — e.g. "6021 CERVINUS RUN"
        ZONING_ZTYPE      — full zone code e.g. "SF-1-NP"
        BASE_ZONE         — base code e.g. "SF-1"
        BASE_ZONE_CATEGORY — e.g. "Single Family Large Lot"
    """
    if not os.path.exists(ZONING_FILE):
        print(f"  Zoning: {ZONING_FILE} not found in project folder.")
        print(f"  Run: python fetch_data.py --only austin_zoning.csv")
        return pd.DataFrame(), {}, {}

    print(f"  Zoning: Loading from {ZONING_FILE}...")
    df = pd.read_csv(ZONING_FILE, low_memory=False)

    required = ["FULL_STREET_NAME", "ZONING_ZTYPE", "BASE_ZONE_CATEGORY"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        print(f"  Zoning: Missing expected columns: {missing}")
        print(f"  Available columns: {df.columns.tolist()}")
        return pd.DataFrame(), {}, {}

    # Build exact match lookup and suffix lookup simultaneously
    lookup = {}   # full street name → (zone_type, zone_category)
    suffix = {}   # street name without block number → (zone_type, zone_category)

    for _, row in df.iterrows():
        street = str(row.get("FULL_STREET_NAME", "") or "").strip().upper()
        ztype  = str(row.get("ZONING_ZTYPE", "Unknown") or "Unknown")
        if not street:
            continue
        val = (ztype, simplify_zone(ztype))
        lookup[street] = val

        # Pre-build suffix index: strip leading block number if present
        parts = street.split()
        if len(parts) > 1 and parts[0].isdigit():
            street_without_block = " ".join(parts[1:])
            if street_without_block not in suffix:
                suffix[street_without_block] = val

    print(f"  Zoning: {len(df):,} records loaded, "
          f"{len(lookup):,} unique streets indexed")
    return df, lookup, suffix


# ── Schools Loader ────────────────────────────────────────────
def _load_schools():
    if not os.path.exists(SCHOOLS_FILE):
        print(f"  Schools: '{SCHOOLS_FILE}' not found.")
        print(f"  Run: python fetch_data.py --only austin_schools.csv")
        return pd.DataFrame()

    print(f"  Schools: Loading from {SCHOOLS_FILE}...")
    try:
        df = pd.read_csv(SCHOOLS_FILE, low_memory=False)
    except Exception as e:
        print(f"  Schools: Failed to load — {e}")
        return pd.DataFrame()

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
    if not os.path.exists(BUS_STOPS_FILE):
        print(f"  Bus Stops: '{BUS_STOPS_FILE}' not found.")
        print(f"  Run: python fetch_data.py --only capmetro_stops.csv")
        return pd.DataFrame()

    print(f"  Bus Stops: Loading from {BUS_STOPS_FILE}...")
    try:
        df = pd.read_csv(BUS_STOPS_FILE)
    except Exception as e:
        print(f"  Bus Stops: Failed to load — {e}")
        return pd.DataFrame()

    if "stop_lat" not in df.columns or "stop_lon" not in df.columns:
        print(f"  Bus Stops: Missing stop_lat/stop_lon columns.")
        print(f"  Columns found: {df.columns.tolist()}")
        return pd.DataFrame()

    df = df.rename(columns={"stop_lat": "lat", "stop_lon": "lon"})
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    df = df[
        df["lat"].between(30.0, 30.7) &
        df["lon"].between(-98.1, -97.4)
    ]

    print(f"  Bus Stops: {len(df):,} stops loaded")
    return df[["lat", "lon"]]


# ── BallTree Builder ──────────────────────────────────────────
def _build_tree(df):
    if df.empty or len(df) < 1:
        return None
    coords = np.radians(df[["lat", "lon"]].values)
    return BallTree(coords, metric="haversine")


# ── Module-level Load ─────────────────────────────────────────
print("Loading land use data...")

_zoning_df, _zoning_lookup, _zoning_suffix = _load_zoning()
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

    FIX: Partial match now uses pre-built _zoning_suffix index
    instead of scanning entire dictionary (O(1) vs O(n)).

    Returns (zone_type, zone_category) or ('Unknown', 'Unknown')
    """
    if not _zoning_lookup:
        return "Unknown", "Unknown"

    # Try road name first — exact match
    if road_name and pd.notna(road_name):
        key = str(road_name).strip().upper()
        if key in _zoning_lookup:
            return _zoning_lookup[key]

        # Try suffix index (street name without block number)
        parts = key.split()
        if len(parts) > 1 and parts[0].isdigit():
            street_without_block = " ".join(parts[1:])
            if street_without_block in _zoning_suffix:
                return _zoning_suffix[street_without_block]

    # Try crash address fallback — exact match
    if crash_address and pd.notna(crash_address):
        key = str(crash_address).strip().upper()
        if key in _zoning_lookup:
            return _zoning_lookup[key]

        # Try suffix index on address too
        parts = key.split()
        if len(parts) > 1 and parts[0].isdigit():
            street_without_block = " ".join(parts[1:])
            if street_without_block in _zoning_suffix:
                return _zoning_suffix[street_without_block]

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


# ── Row-wise Wrapper (unused — kept for Phase 2 df.apply use) ─
def process_land_use(row):
    """Row-wise wrapper for use with df.apply() in Phase 2"""
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
        print("  Run: python fetch_data.py")
    print("───────────────────────────────────────────────────\n")


# Call verify automatically so you always see the summary on load
verify_data()