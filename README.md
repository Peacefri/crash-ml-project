# crash-ml-project

Austin crash enrichment pipeline combining four external data integrations:

- Weather: Open-Meteo historical hourly API, cached in `weather_cache.json`.
- Roads: OpenStreetMap through OSMnx, cached in `austin_road_network.pkl`.
- Traffic exposure: TxDOT AADT stations in `txdot_aadt.csv`, with historical-year lookup and road-type estimates as fallback.
- Land use: Austin zoning plus school and CapMetro stop CSVs, matched by address and geographic proximity.

## Run

1. Install dependencies: `.venv/Scripts/python.exe -m pip install -r requirements.txt`
2. Put the input crash file in the project folder. The default is `crashes_with_weather.csv`; override it with `CRASH_INPUT_FILE`.
3. Download the external datasets: `.venv/Scripts/python.exe fetch_data.py`. This writes `txdot_aadt.csv`, `austin_zoning.csv`, `austin_schools.csv`, and `capmetro_stops.csv`. They are gitignored because of their size, so a fresh clone has none of them — this step is what puts them back. Missing local files are reported and do not prevent weather or road-type fallback processing.
4. Run: `.venv/Scripts/python.exe main.py`

The output is `crashes_final_enriched.csv`. Each integration writes source/proximity fields so coverage can be audited, especially `AADT_Source`, `Weather_Code`, `Road_Type_Label`, `Zone_Category`, `Near_School`, and `Near_Bus_Stop`.

## Integration checklist

Before a full run, verify that the crash input contains latitude, longitude, crash timestamp, and address/street fields. Confirm the four local datasets contain the columns documented in the corresponding `*_data.py` module. The first road run needs network access to download Austin's OSM graph; later runs use the cache. Open-Meteo also needs network access unless all requested weather records are already cached.

## Data sources

| File | Source | Notes |
| --- | --- | --- |
| `txdot_aadt.csv` | [TxDOT AADT Annuals (Public View)](https://gis-txdot.opendata.arcgis.com/datasets/txdot-aadt-annuals) | 1,585 active Austin-metro stations, 19 years of history |
| `austin_zoning.csv` | [Zoning By Address, dataset `nbzi-qabm`](https://data.austintexas.gov/Building-and-Development/Zoning-By-Address/nbzi-qabm) | ~263k addresses, ~22 MB |
| `austin_schools.csv` | [NCES Public School Locations - Current](https://data-nces.opendata.arcgis.com/) | 649 schools in the five Austin-metro counties |
| `capmetro_stops.csv` | [CapMetro GTFS, dataset `r4v4-vz24`](https://data.texas.gov/dataset/CapMetro-GTFS/r4v4-vz24) | `stops.txt` from the GTFS zip, ~2,364 stops |

Schools come from NCES rather than the City of Austin's "Schools with Data" set (`63ig-4knr`) that the module header originally named: that dataset publishes no coordinate columns, so it cannot support a distance calculation. NCES covers public and charter schools; private schools are not included.

## Known limitations

AADT estimates are used when no same-road TxDOT station is found. In practice this is most crashes: TxDOT records state highways under route codes (`IH0035`, `US0183`) while OSM supplies names (`Interstate 35`), so only city-street stations match by name.

`roads_match()` compares leftover words after stripping directional prefixes, but not street-type suffixes, so any two roads sharing `BLVD`/`ST`/`RD` within the 1 km radius count as a match — `MARTIN LUTHER KING JR BLVD` matches a station on `AIRPORT BLVD`.

Zoning is address-based, so unmatched or differently formatted addresses receive `Unknown`; on the current 99-row sample 43 of 99 crashes resolve to a real zone. Missing OSM `lit` tags remain null rather than being treated as unlit.

Console output uses box-drawing characters, which crash on a cp1252 Windows console. Run with `PYTHONIOENCODING=utf-8` (or `chcp 65001`) if you hit `UnicodeEncodeError`.
