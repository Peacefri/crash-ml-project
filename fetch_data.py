# ============================================================
# fetch_data.py — Austin Crash Safety Prediction System
# Downloads the four external datasets the pipeline needs.
#
# These files are in .gitignore (too large for the repo), so a
# fresh clone has none of them. Run this once before main.py:
#
#     .venv/Scripts/python.exe fetch_data.py
#
# Produces, in the project folder:
#   crashes_raw.csv     — Austin crash records (main.py input)
#   txdot_aadt.csv      — TxDOT AADT stations, Austin metro
#   austin_zoning.csv   — Austin Zoning By Address
#   austin_schools.csv  — NCES school locations, Austin metro
#   capmetro_stops.csv  — CapMetro GTFS stops.txt
#
# Each dataset is fetched independently; one failure does not
# stop the others. Existing files are skipped unless --force.
# ============================================================

import argparse
import io
import os
import sys
import zipfile

import pandas as pd
import requests

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Counties matching AUSTIN_COUNTIES in aadt_data.py
AUSTIN_COUNTIES = ["Travis", "Williamson", "Hays", "Bastrop", "Caldwell"]

TIMEOUT = 120


# ── Austin crash records ─────────────────────────────────────
# Socrata dataset y2wy-tgr5, "Austin Crash Report Data - Crash
# Level Records" on the Austin Open Data Portal:
# https://data.austintexas.gov/Transportation-and-Mobility/Austin-Crash-Report-Data-Crash-Level-Records/y2wy-tgr5
#
# The full set is ~230k geocoded crashes. We pull a small slice so
# the pipeline can be exercised end to end quickly. The default is
# the oldest records for reproducible tests; --latest requests the
# newest records for a data refresh.
CRASH_URL   = "https://data.austintexas.gov/resource/y2wy-tgr5.json"
CRASH_LIMIT = 100          # override with --crashes N
CRASH_PAGE  = 1000         # Socrata page size when N is large

# Only rows the enrichment pipeline can actually use: every stage
# (road, AADT, weather, land use) is keyed off coordinates, and a
# missing timestamp makes the time features unusable.
CRASH_WHERE = (
    "latitude IS NOT NULL AND longitude IS NOT NULL "
    "AND crash_timestamp_ct IS NOT NULL "
    "AND is_deleted = false"
)

# The SODA API returns machine field names; main.py and the chart
# scripts expect the portal's human-readable header names. Only the
# fields whose two names differ need mapping — the rest are identical.
CRASH_RENAME = {
    "id":                         "ID",
    "cris_crash_id":              "Crash ID",
    "crash_timestamp_ct":         "Crash timestamp (US/Central)",
    "crash_timestamp":            "Crash timestamp",
    "is_deleted":                 "Is deleted",
    "is_temp_record":             "Is temporary record",
    "law_enf_fatality_count":     "Law enforcement fatality count",
    "rpt_street_pfx":             "Reported street prefix",
    "est_comp_cost_crash_based":  "Estimated Maximum Comprehensive Cost",
    "est_total_person_comp_cost": "Estimated Total Comprehensive Cost",
    "location_id":                "Location ID",
    "location_group":             "Location group",
    "address_display":            "Address",
    "collsn_desc":                "Collision type",
}

# Written in this order so the file matches the portal's own CSV export.
CRASH_COLUMNS = [
    "ID", "Crash ID", "crash_fatal_fl", "case_id", "rpt_block_num",
    "rpt_street_name", "rpt_street_sfx", "crash_speed_limit",
    "road_constr_zone_fl", "latitude", "longitude", "crash_sev_id",
    "sus_serious_injry_cnt", "nonincap_injry_cnt", "poss_injry_cnt",
    "non_injry_cnt", "unkn_injry_cnt", "tot_injry_cnt", "death_cnt",
    "units_involved", "point", "motor_vehicle_death_count",
    "motor_vehicle_serious_injury_count", "bicycle_death_count",
    "bicycle_serious_injury_count", "pedestrian_death_count",
    "pedestrian_serious_injury_count", "motorcycle_death_count",
    "motorcycle_serious_injury_count", "other_death_count",
    "other_serious_injury_count", "onsys_fl", "private_dr_fl",
    "micromobility_serious_injury_count", "micromobility_death_count",
    "Crash timestamp (US/Central)", "Crash timestamp",
    "Is deleted", "Is temporary record", "Law enforcement fatality count",
    "Reported street prefix", "Estimated Maximum Comprehensive Cost",
    "Estimated Total Comprehensive Cost", "Location ID",
    "Location group", "Address", "Collision type",
]


def fetch_crashes(dest, limit=CRASH_LIMIT, newest=False):
    """Pull a slice of geocoded crashes, oldest by default or newest."""
    rows = []
    while len(rows) < limit:
        resp = requests.get(
            CRASH_URL,
            params={
                "$where":  CRASH_WHERE,
                "$order":  "crash_timestamp_ct DESC" if newest else "crash_timestamp_ct ASC",
                "$limit":  min(CRASH_PAGE, limit - len(rows)),
                "$offset": len(rows),
            },
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        page = resp.json()
        if not page:
            break                      # dataset exhausted before `limit`
        rows.extend(page)
        if limit > CRASH_PAGE:
            print(f"    ...{len(rows):,} crashes")

    if not rows:
        raise RuntimeError("No crashes returned — check the SoQL filter.")

    df = pd.DataFrame(rows).rename(columns=CRASH_RENAME)

    # Socrata returns `point` as a GeoJSON dict; the portal's CSV export
    # renders it as WKT. Match the export so downstream code sees one shape.
    if "point" in df.columns:
        df["point"] = df["point"].apply(
            lambda p: (
                f"POINT ({p['coordinates'][0]} {p['coordinates'][1]})"
                if isinstance(p, dict) and p.get("coordinates") else None
            )
        )

    # Drop Socrata's :@computed_region_* columns and enforce column order.
    # Any expected-but-absent column is created empty rather than silently
    # shifting the rest — main.py validates against these names.
    for col in CRASH_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[CRASH_COLUMNS]

    df.to_csv(dest, index=False)

    ts = pd.to_datetime(df["Crash timestamp (US/Central)"], errors="coerce")
    return (
        f"{len(df):,} crashes, "
        f"{ts.min():%Y-%m-%d} to {ts.max():%Y-%m-%d}, "
        f"{df['crash_sev_id'].nunique()} severity levels"
    )


# ── TxDOT AADT ───────────────────────────────────────────────
# ArcGIS FeatureServer behind the "TxDOT AADT Annuals" dataset on
# the TxDOT Open Data Portal. Field names match aadt_data.py
# exactly, including the 19 AADT_RPT_HIST_nn_QTY history columns.
AADT_URL = (
    "https://services.arcgis.com/KTcxiTD9dsQw4r7Z/arcgis/rest/services/"
    "TxDOT_AADT_Annuals_(Public_View)/FeatureServer/0/query"
)
AADT_PAGE_SIZE = 2000   # layer's maxRecordCount


def fetch_aadt(dest):
    """Page through the TxDOT AADT layer, Austin metro counties only."""
    counties = ",".join(f"'{c}'" for c in AUSTIN_COUNTIES)
    where = f"CNTY_NM IN ({counties}) AND ACTIVE = 1"

    rows = []
    offset = 0
    while True:
        resp = requests.get(
            AADT_URL,
            params={
                "where":            where,
                "outFields":        "*",
                "returnGeometry":   "false",
                "resultOffset":     offset,
                "resultRecordCount": AADT_PAGE_SIZE,
                "f":                "json",
            },
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json()

        if "error" in payload:
            raise RuntimeError(f"ArcGIS error: {payload['error']}")

        features = payload.get("features", [])
        if not features:
            break

        rows.extend(f["attributes"] for f in features)
        print(f"    ...{len(rows):,} stations")

        # exceededTransferLimit absent means this was the last page
        if not payload.get("exceededTransferLimit"):
            break
        offset += len(features)

    if not rows:
        raise RuntimeError("No AADT stations returned — check county names.")

    df = pd.DataFrame(rows)
    df.to_csv(dest, index=False)
    return (
        f"{len(df):,} stations, "
        f"{df['ON_ROAD'].nunique():,} roads, "
        f"latest report year {int(df['AADT_RPT_YEAR'].max())}"
    )


# ── Austin Zoning By Address ─────────────────────────────────
# Socrata dataset nbzi-qabm. Columns FULL_STREET_NAME,
# ZONING_ZTYPE, BASE_ZONE, BASE_ZONE_CATEGORY — as land_use_data.py
# expects. ~50MB, so stream it to disk rather than buffering.
ZONING_URL = (
    "https://data.austintexas.gov/api/views/nbzi-qabm/rows.csv"
    "?accessType=DOWNLOAD"
)


def fetch_zoning(dest):
    with requests.get(ZONING_URL, stream=True, timeout=TIMEOUT) as resp:
        resp.raise_for_status()
        written = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                written += len(chunk)
                print(f"    ...{written / 1e6:.0f} MB", end="\r")
    print(" " * 40, end="\r")

    df = pd.read_csv(dest, low_memory=False)
    missing = [
        c for c in ("FULL_STREET_NAME", "ZONING_ZTYPE", "BASE_ZONE_CATEGORY")
        if c not in df.columns
    ]
    if missing:
        raise RuntimeError(f"Zoning file missing expected columns: {missing}")

    return (
        f"{len(df):,} addresses, "
        f"{df['FULL_STREET_NAME'].nunique():,} unique streets"
    )


# ── Schools ──────────────────────────────────────────────────
# NOTE: the City of Austin "Schools with Data" set (63ig-4knr)
# referenced in land_use_data.py has NO coordinate columns, so it
# cannot drive a distance calculation. NCES Public School Locations
# carries LAT/LON directly, which is what _load_schools() looks for.
# Covers public and charter schools; private schools are not included.
SCHOOLS_URL = (
    "https://services1.arcgis.com/Ua5sjt3LWTPigjyD/arcgis/rest/services/"
    "Public_School_Locations_Current/FeatureServer/0/query"
)


def fetch_schools(dest):
    counties = ",".join(f"'{c} County'" for c in AUSTIN_COUNTIES)
    resp = requests.get(
        SCHOOLS_URL,
        params={
            "where":          f"STATE = 'TX' AND NMCNTY IN ({counties})",
            "outFields":      "NAME,STREET,CITY,ZIP,NMCNTY,LAT,LON",
            "returnGeometry": "false",
            "f":              "json",
        },
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    payload = resp.json()

    if "error" in payload:
        raise RuntimeError(f"ArcGIS error: {payload['error']}")

    rows = [f["attributes"] for f in payload.get("features", [])]
    if not rows:
        raise RuntimeError("No schools returned — check county names.")

    df = pd.DataFrame(rows)
    df.to_csv(dest, index=False)
    return f"{len(df):,} schools across {df['NMCNTY'].nunique()} counties"


# ── CapMetro bus stops ───────────────────────────────────────
# The GTFS feed is published as a zip blob on data.texas.gov.
# We pull stops.txt out of it and save it under the name
# land_use_data.py expects.
CAPMETRO_URL = (
    "https://data.texas.gov/api/views/r4v4-vz24/files/"
    "f86ee316-d682-4518-aae4-d683f8296229?filename=capmetro.zip"
)


def fetch_capmetro(dest):
    resp = requests.get(CAPMETRO_URL, timeout=TIMEOUT)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        if "stops.txt" not in z.namelist():
            raise RuntimeError(
                f"stops.txt not in GTFS zip; found {z.namelist()}"
            )
        df = pd.read_csv(io.BytesIO(z.read("stops.txt")))

        feed_version = ""
        if "feed_info.txt" in z.namelist():
            info = pd.read_csv(io.BytesIO(z.read("feed_info.txt")))
            if "feed_start_date" in info.columns and len(info):
                feed_version = f", feed starts {info['feed_start_date'].iloc[0]}"

    if "stop_lat" not in df.columns or "stop_lon" not in df.columns:
        raise RuntimeError(f"stops.txt missing stop_lat/stop_lon: {list(df.columns)}")

    df.to_csv(dest, index=False)
    return f"{len(df):,} stops{feed_version}"


# ── Driver ───────────────────────────────────────────────────
DATASETS = [
    ("Austin crashes",   "crashes_raw.csv",    fetch_crashes),
    ("TxDOT AADT",       "txdot_aadt.csv",     fetch_aadt),
    ("Austin zoning",    "austin_zoning.csv",  fetch_zoning),
    ("Schools",          "austin_schools.csv", fetch_schools),
    ("CapMetro stops",   "capmetro_stops.csv", fetch_capmetro),
]


def main():
    parser = argparse.ArgumentParser(
        description="Download the external datasets used by main.py"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="re-download datasets even if the file already exists"
    )
    parser.add_argument(
        "--only", metavar="FILENAME", action="append",
        help="fetch only this output file (repeatable)"
    )
    parser.add_argument(
        "--crashes", type=int, default=CRASH_LIMIT, metavar="N",
        help=f"how many crash records to pull (default {CRASH_LIMIT})"
    )
    parser.add_argument(
        "--latest", action="store_true",
        help="when fetching crashes, use the newest records instead of the oldest"
    )
    args = parser.parse_args()

    targets = DATASETS
    if args.only:
        targets = [d for d in DATASETS if d[1] in args.only]
        if not targets:
            print(f"No dataset matches {args.only}.")
            print(f"Valid names: {[d[1] for d in DATASETS]}")
            return 1

    print("Fetching external datasets into:")
    print(f"  {PROJECT_DIR}\n")

    failures = []
    for label, filename, fetch in targets:
        dest = os.path.join(PROJECT_DIR, filename)

        if os.path.exists(dest) and not args.force:
            size = os.path.getsize(dest) / 1e6
            print(f"  [skip] {label:<16} {filename} already exists "
                  f"({size:.1f} MB) - use --force to refresh")
            continue

        print(f"  [get ] {label:<16} -> {filename}")
        try:
            if fetch is fetch_crashes:
                summary = fetch(dest, limit=args.crashes, newest=args.latest)
            else:
                summary = fetch(dest)
            print(f"         {summary}")
        except Exception as exc:
            # Leave no truncated file behind for main.py to trip over
            if os.path.exists(dest):
                os.remove(dest)
            print(f"         FAILED: {exc}")
            failures.append(label)

    print()
    if failures:
        print(f"Finished with errors. Failed: {', '.join(failures)}")
        print("Re-run to retry; other datasets were saved.")
        return 1

    print("All datasets ready. Next: .venv/Scripts/python.exe main.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
