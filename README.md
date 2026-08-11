# crash-ml-project

Austin crash enrichment pipeline combining four external data integrations:

- Weather: Open-Meteo historical hourly API, cached in `weather_cache.json`.
- Roads: OpenStreetMap through OSMnx, cached in `austin_road_network.pkl`.
- Traffic exposure: TxDOT AADT stations in `txdot_aadt.csv`, with historical-year lookup and road-type estimates as fallback.
- Land use: Austin zoning plus school and CapMetro stop CSVs, matched by address and geographic proximity.

## Run

1. Install dependencies: `.venv/Scripts/python.exe -m pip install -r requirements.txt`
2. Download the datasets: `.venv/Scripts/python.exe fetch_data.py`. This writes `crashes_raw.csv`, `txdot_aadt.csv`, `austin_zoning.csv`, `austin_schools.csv`, and `capmetro_stops.csv`. They are gitignored, so a fresh clone has none of them — this step is what puts them back. Missing local files are reported and do not prevent weather or road-type fallback processing.
3. Run: `.venv/Scripts/python.exe main.py`

`crashes_raw.csv` holds the crash records the pipeline enriches. By default `fetch_data.py` pulls the **oldest 100** geocoded crashes for a reproducible test. To refresh with current records, run `.venv/Scripts/python.exe fetch_data.py --force --latest --crashes 100`, then run `.venv/Scripts/python.exe main.py`. Pull more with `--crashes N`, or point `CRASH_INPUT_FILE` at a different file. Rows without coordinates or a timestamp are filtered out at fetch time, since every enrichment stage is keyed off them.

The output is `crashes_final_enriched.csv`. Each integration writes source/proximity fields so coverage can be audited, especially `AADT_Source`, `Weather_Code`, `Road_Type_Label`, `Zone_Category`, `Near_School`, and `Near_Bus_Stop`.

## Integration checklist

Before a full run, verify that the crash input contains latitude, longitude, crash timestamp, and address/street fields. Confirm the four local datasets contain the columns documented in the corresponding `*_data.py` module. The first road run needs network access to download Austin's OSM graph; later runs use the cache. Open-Meteo also needs network access unless all requested weather records are already cached.

## Data sources

### CSV Files (downloaded and saved locally)

| Dataset | Type | Source | Notes |
| --- | --- | --- | --- |
| `crashes_raw.csv` | CSV | [Austin Crash Report Data, dataset `y2wy-tgr5`](https://data.austintexas.gov/Transportation-and-Mobility/Austin-Crash-Report-Data-Crash-Level-Records/y2wy-tgr5/about_data) | ~230k geocoded crashes available; oldest 100 fetched by default |
| `txdot_aadt.csv` | CSV | [TxDOT AADT Annuals (ArcGIS)](https://gis-txdot.opendata.arcgis.com/datasets/d5f56ecd2b274b4d8dc3c2d6fe067d37_0/explore) | 1,585 active Austin-metro stations, 19 years of Annual Average Daily Traffic |
| `austin_zoning.csv` | CSV | [Austin Zoning By Address, dataset `nbzi-qabm`](https://data.austintexas.gov/Building-and-Development/Zoning-By-Address/nbzi-qabm) | ~263k addresses with zoning classification |
| `austin_schools.csv` | CSV | [NCES Public School Locations](https://services1.arcgis.com/Ua5sjt3LWTPigjyD/arcgis/rest/services/Public_School_Locations_Current/FeatureServer) | 649 public and charter schools in five Austin-metro counties |
| `capmetro_stops.csv` | CSV | [CapMetro GTFS, dataset `r4v4-vz24`](https://data.texas.gov/dataset/CapMetro-GTFS/r4v4-vz24) | General Transit Feed Specification: ~2,364 bus stops |

### APIs (queried on-demand, not saved locally)

| Source | Type | Documentation | Notes |
| --- | --- | --- | --- |
| **Weather** | API | [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) | Free hourly historical data: temperature, precipitation, wind, visibility, weather codes (no API key required); cached in `weather_cache.json` |
| **Roads** | API | [OpenStreetMap via OSMnx](https://osmnx.readthedocs.io/) | Road network, road types, lanes, speed limits, intersections, lighting, curvature; cached in `austin_road_network.pkl` |

Schools come from NCES rather than the City of Austin's "Schools with Data" set (`63ig-4knr`) that the module header originally named: that dataset publishes no coordinate columns, so it cannot support a distance calculation. NCES covers public and charter schools; private schools are not included.

## Known limitations

AADT estimates are used when no same-road TxDOT station is found. In practice this is most crashes: TxDOT records state highways under route codes (`IH0035`, `US0183`) while OSM supplies names (`Interstate 35`), so only city-street stations match by name.

`roads_match()` compares leftover words after stripping directional prefixes, but not street-type suffixes, so any two roads sharing `BLVD`/`ST`/`RD` within the 1 km radius count as a match — `MARTIN LUTHER KING JR BLVD` matches a station on `AIRPORT BLVD`.

Zoning is address-based, so unmatched or differently formatted addresses receive `Unknown`; on the current 99-row sample 43 of 99 crashes resolve to a real zone. Missing OSM `lit` tags remain null rather than being treated as unlit.

Console output uses box-drawing characters, which crash on a cp1252 Windows console. Run with `PYTHONIOENCODING=utf-8` (or `chcp 65001`) if you hit `UnicodeEncodeError`.
