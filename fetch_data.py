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
