# ============================================================
# crash_frequency_map.py — Austin Crash Safety Prediction System
#
# Map 1 — crash_individual_map.html
#   SHAPE  = time of day  (triangle=night, circle=day)
#   BORDER = full weather condition color (Rain, Fog, Clear etc)
#            Unknown condition markers are hidden entirely
#   COLOR  = crash severity (0-5)
#   RINGS  = magenta/pink — repeated crash hotspots
#
# Map 2 — crash_hotspot_map.html
#   Grid frequency + top 10 hotspot ranking (trails excluded)
#
# Run locally:  python crash_frequency_map.py
# ============================================================

import pandas as pd
import numpy as np
import folium
import branca.colormap as cm

INPUT_FILE     = "crashes_final_enriched.csv"
SCHOOLS_FILE   = "austin_schools.csv"
BUS_STOPS_FILE = "capmetro_stops.csv"

# ── Zone category colors ──────────────────────────────────────
ZONE_COLORS = {
    "Residential":    "#228B22",   # Green
    "Commercial":     "#FF6600",   # Orange
    "Industrial":     "#808080",   # Gray
    "Mixed Use":      "#9400D3",   # Purple
    "Civic / Public": "#2E75B6",   # Blue
    "Other":          "#A9A9A9",   # Light gray
    "Unknown":        "#DDDDDD",   # Very light gray
}

SEV_COLORS = {
    0: "#808080",
    1: "#FF6600",
    2: "#FFD700",
    3: "#90EE90",
    4: "#CC0000",
    5: "#228B22",
}
SEV_LABELS = {
    0: "Unknown",
    1: "Incapacitating Injury",
    2: "Non-Incapacitating Injury",
    3: "Possible Injury",
    4: "Killed",
    5: "Not Injured"
}

# ── Weather condition border colors ──────────────────────────
# Each weather condition gets a unique border color on the marker.
# Unknown condition markers are skipped entirely — not shown.
WEATHER_COLORS = {
    "Clear":                  "#87CEEB",   # Sky blue
    "Partly Cloudy":          "#B0C4DE",   # Steel blue
    "Drizzle":                "#4682B4",   # Medium blue
    "Rain":                   "#0000CD",   # Dark blue
    "Rain Showers":           "#1E90FF",   # Dodger blue
    "Thunderstorm":           "#8B008B",   # Dark magenta
    "Foggy":                  "#708090",   # Slate gray
    "Snow":                   "#00CED1",   # Turquoise
    "Other":                  "#A9A9A9",   # Dark gray
}
# Unknown weather = skip marker entirely (not shown on map)
SKIP_WEATHER = {"Unknown", None, ""}


# ── FIX 2: Road types that are NOT roads — exclude from ranking
# These are trails, paths, and non-vehicular ways that OSMnx
# sometimes snaps to when the actual road is nearby
NON_ROAD_TYPES = {
    "path", "track", "footway", "cycleway",
    "pedestrian", "steps", "bridleway"
}

# FIX 2: Keywords in road names that indicate a trail not a road
TRAIL_KEYWORDS = [
    "trail", "trailhead", "greenway", "greenbelt",
    "hike", "bike path", "creek path"
]


def safe_int(val):
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def safe_mode(x):
    vals = x.dropna()
    if len(vals) == 0:
        return "Unknown"
    m = vals.mode()
    return m.iloc[0] if len(m) > 0 else "Unknown"


def is_trail(road_name, highway_type):
    """Return True if this looks like a trail rather than a road."""
    if pd.isna(road_name) or road_name is None:
        return False
    name_lower = str(road_name).lower()
    for kw in TRAIL_KEYWORDS:
        if kw in name_lower:
            return True
    if str(highway_type).lower() in NON_ROAD_TYPES:
        return True
    return False


def load_data():
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} crash records")

    # FIX 2: Flag trail crashes so we can exclude from ranking
    df["Is_Trail"] = df.apply(
        lambda r: is_trail(r.get("Road_Name"), r.get("Highway_Type")),
        axis=1
    )
    trail_count = df["Is_Trail"].sum()
    if trail_count > 0:
        print(f"  Flagged {trail_count} crashes near trails/paths "
              f"(excluded from hotspot ranking)")
    return df


def make_icon(shape, fill_color, border_color, dot_size=22):
    """
    Build a folium DivIcon using SVG shapes.
    dot_size controls the marker size — larger = more dangerous lighting.
    shape = 'triangle' for nighttime, 'circle' for daytime.
    """
    size = dot_size
    bw   = 3

    if shape == "triangle":
        pts = f"{size//2},2 {size-2},{size-2} 2,{size-2}"
        svg = (
            f'<svg width="{size}" height="{size}" '
            f'xmlns="http://www.w3.org/2000/svg">'
            f'<polygon points="{pts}" '
            f'fill="{fill_color}" '
            f'stroke="{border_color}" '
            f'stroke-width="{bw}" '
            f'stroke-linejoin="round"/>'
            f'</svg>'
        )
    else:
        r  = size // 2 - 2
        cx = cy = size // 2
        svg = (
            f'<svg width="{size}" height="{size}" '
            f'xmlns="http://www.w3.org/2000/svg">'
            f'<circle cx="{cx}" cy="{cy}" r="{r}" '
            f'fill="{fill_color}" '
            f'stroke="{border_color}" '
            f'stroke-width="{bw}"/>'
            f'</svg>'
        )

    return folium.DivIcon(
        html=svg,
        icon_size=(size, size),
        icon_anchor=(size // 2, size // 2)
    )


# ── Land Use Layer Builder ────────────────────────────────────
def add_land_use_layers(m, df):
    """
    Adds three toggleable layers to a folium map:

    1. Zone Color Layer — circles colored by zone category
       (Residential=green, Commercial=orange, Industrial=gray,
        Mixed Use=purple, Civic=blue)
       Only shown if Zone_Category column exists in df.

    2. Schools Layer — school emoji markers at every school
       loaded from austin_schools.csv

    3. Bus Stops Layer — bus emoji markers at every stop
       loaded from capmetro_stops.csv

    All three are separate FeatureGroups so they can be
    toggled on/off independently via the layer control.
    """
    import os

    # ── Layer 1: Zone Color Circles ───────────────────────────
    zone_layer = folium.FeatureGroup(
        name="&#127963; Zone Categories", show=False
    )
    if "Zone_Category" in df.columns:
        for _, row in df.iterrows():
            zone = str(row.get("Zone_Category", "Unknown") or "Unknown")
            color = ZONE_COLORS.get(zone, "#DDDDDD")
            folium.Circle(
                location=[row["latitude"], row["longitude"]],
                radius=80,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.25,
                weight=1,
                tooltip=f"Zone: {zone}",
                popup=folium.Popup(
                    f"<b>Zone Category:</b> {zone}<br>"
                    f"<b>Zone Type:</b> "
                    f"{str(row.get('Zone_Type','Unknown') or 'Unknown')}",
                    max_width=200
                )
            ).add_to(zone_layer)
    else:
        # Zone data not yet in CSV — show placeholder message
        folium.Marker(
            location=[30.2672, -97.7431],
            icon=folium.DivIcon(
                html='<div style="background:white;padding:8px;'
                     'border:2px solid orange;border-radius:6px;'
                     'font-size:11px;font-family:Arial;">'
                     '&#9888; Zone data not yet in CSV.<br>'
                     'Re-run main.py to add land use columns.</div>',
                icon_size=(260, 50), icon_anchor=(130, 25)
            )
        ).add_to(zone_layer)
    zone_layer.add_to(m)

    # ── Layer 2: Schools ──────────────────────────────────────
    schools_layer = folium.FeatureGroup(
        name="&#127979; Schools", show=False
    )
    if os.path.exists(SCHOOLS_FILE):
        try:
            schools = pd.read_csv(SCHOOLS_FILE)
            lat_col = next(
                (c for c in schools.columns if "lat" in c.lower()), None
            )
            lon_col = next(
                (c for c in schools.columns if "lon" in c.lower()), None
            )
            name_col = next(
                (c for c in schools.columns
                 if "name" in c.lower() or "campname" in c.lower()), None
            )
            if lat_col and lon_col:
                schools[lat_col] = pd.to_numeric(
                    schools[lat_col], errors="coerce"
                )
                schools[lon_col] = pd.to_numeric(
                    schools[lon_col], errors="coerce"
                )
                schools = schools.dropna(subset=[lat_col, lon_col])
                schools = schools[
                    schools[lat_col].between(30.0, 30.7) &
                    schools[lon_col].between(-98.1, -97.4)
                ]
                for _, s in schools.iterrows():
                    sname = str(
                        s.get(name_col, "School") if name_col else "School"
                    )
                    folium.Marker(
                        location=[s[lat_col], s[lon_col]],
                        icon=folium.DivIcon(
                            html=(
                                '<div style="font-size:18px;'
                                'text-align:center;line-height:1;">'
                                '&#127979;</div>'
                            ),
                            icon_size=(24, 24),
                            icon_anchor=(12, 12)
                        ),
                        tooltip=f"School: {sname}",
                        popup=folium.Popup(
                            f"<b>&#127979; {sname}</b>",
                            max_width=200
                        )
                    ).add_to(schools_layer)
        except Exception as e:
            print(f"  Schools layer error: {e}")
    schools_layer.add_to(m)

    # ── Layer 3: Bus Stops ────────────────────────────────────
    bus_layer = folium.FeatureGroup(
        name="&#128652; Bus Stops", show=False
    )
    if os.path.exists(BUS_STOPS_FILE):
        try:
            stops = pd.read_csv(BUS_STOPS_FILE)
            if "stop_lat" in stops.columns and "stop_lon" in stops.columns:
                stops["stop_lat"] = pd.to_numeric(
                    stops["stop_lat"], errors="coerce"
                )
                stops["stop_lon"] = pd.to_numeric(
                    stops["stop_lon"], errors="coerce"
                )
                stops = stops.dropna(subset=["stop_lat", "stop_lon"])
                stops = stops[
                    stops["stop_lat"].between(30.0, 30.7) &
                    stops["stop_lon"].between(-98.1, -97.4)
                ]
                for _, s in stops.iterrows():
                    sname = str(s.get("stop_name", "Bus Stop") or "Bus Stop")
                    folium.Marker(
                        location=[s["stop_lat"], s["stop_lon"]],
                        icon=folium.DivIcon(
                            html=(
                                '<div style="font-size:14px;'
                                'text-align:center;line-height:1;">'
                                '&#128652;</div>'
                            ),
                            icon_size=(20, 20),
                            icon_anchor=(10, 10)
                        ),
                        tooltip=f"Bus Stop: {sname}",
                        popup=folium.Popup(
                            f"<b>&#128652; {sname}</b>",
                            max_width=200
                        )
                    ).add_to(bus_layer)
        except Exception as e:
            print(f"  Bus stops layer error: {e}")
    bus_layer.add_to(m)

    return m


# ── Map 1: Individual Crash Map ───────────────────────────────
def build_individual_map(df):
    print("\nBuilding Map 1 — Individual crash map...")

    # FIX 4: Tighter zoom centered on Austin city center
    m = folium.Map(
        location=[30.2672, -97.7431],
        zoom_start=12,
        tiles="CartoDB positron"
    )

    # Hotspot rings
    df["lat_r"] = df["latitude"].round(3)
    df["lon_r"] = df["longitude"].round(3)
    loc_counts  = df.groupby(["lat_r", "lon_r"]).size().reset_index(name="n")

    # Ring colors are MAGENTA/PINK — completely absent from the
    # severity palette so zero confusion. Made larger so they
    # dominate the map visually and signal problem areas clearly.
    RING_COLORS = {
        "high":   "#FF00FF",   # Bright magenta = 5+ crashes
        "medium": "#C71585",   # Deep pink/rose = 3-4 crashes
        "low":    "#FF69B4",   # Hot pink       = 2 crashes
    }
    ring_layer = folium.FeatureGroup(
        name="Hotspot Rings (2+ crashes same location)"
    )
    for _, spot in loc_counts[loc_counts["n"] >= 2].iterrows():
        n = int(spot["n"])
        if n >= 5:
            color   = RING_COLORS["high"]
            radius  = 350        # Very large — dominant on map
            weight  = 5
            opacity = 0.25
            label   = f"MAJOR HOTSPOT: {n} crashes at this location"
        elif n >= 3:
            color   = RING_COLORS["medium"]
            radius  = 220
            weight  = 4
            opacity = 0.20
            label   = f"Hotspot: {n} crashes at this location"
        else:
            color   = RING_COLORS["low"]
            radius  = 130
            weight  = 3
            opacity = 0.15
            label   = f"Repeated: {n} crashes at this location"
        folium.Circle(
            location=[spot["lat_r"], spot["lon_r"]],
            radius=radius,
            color=color, fill=True, fill_color=color,
            fill_opacity=opacity, weight=weight,
            tooltip=label,
            popup=folium.Popup(
                f"<div style='font-family:Arial;font-size:12px;'>"
                f"<div style='background:#C71585;color:white;"
                f"padding:6px 10px;margin:-14px -20px 10px -20px;"
                f"border-radius:6px 6px 0 0;font-weight:bold;'>"
                f"&#128308; {label}</div>"
                f"<b>What this ring means:</b><br>"
                f"This is NOT a single crash. This pink/magenta ring "
                f"marks a location where <b>{n} separate crashes</b> "
                f"have occurred within ~100 meters of each other "
                f"over time. This signals a <b>repeated pattern</b> "
                f"at this intersection or road segment that needs "
                f"safety attention.<br><br>"
                f"<span style='font-size:11px;color:#666;'>"
                f"Coordinates: {spot['lat_r']:.3f}, "
                f"{spot['lon_r']:.3f}</span>"
                f"</div>",
                max_width=280
            )
        ).add_to(ring_layer)
    ring_layer.add_to(m)

    # Crash markers — one FeatureGroup per severity for filtering
    # FIX 3: Separate layer per severity so toggle buttons work
    sev_layers = {}
    for sev_code, sev_label in SEV_LABELS.items():
        layer = folium.FeatureGroup(
            name=f"Severity {sev_code}: {sev_label}",
            show=True
        )
        sev_layers[sev_code] = layer

    for _, row in df.iterrows():
        try:
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            sev = safe_int(row.get("crash_sev_id", 0))

            fill_color  = SEV_COLORS.get(sev, "#808080")
            sev_label   = SEV_LABELS.get(sev, "Unknown")

            # ── Weather condition → border color ──────────────
            # Full weather condition replaces wet/dry binary flag
            # Unknown weather crashes are skipped — not shown
            weather   = str(row.get("Weather_Condition", "") or "")
            if weather in SKIP_WEATHER or weather not in WEATHER_COLORS:
                continue   # Skip unknown weather crashes entirely
            border_color = WEATHER_COLORS[weather]
            cond_text    = weather

            is_dark   = safe_int(row.get("Is_Dark", 0))
            shape     = "triangle" if is_dark == 1 else "circle"
            time_text = "Nighttime (8pm–6am)" if is_dark == 1 \
                        else "Daytime (6am–8pm)"

            aadt      = row.get("AADT", None)
            aadt_src  = str(row.get("AADT_Source", "unknown"))
            hour      = row.get("Crash Hour", "N/A")
            speed     = str(row.get("Speed_Limit", "N/A") or "N/A")
            curvature = row.get("Road_Curvature", None)
            road      = str(row.get("Road_Type_Label", "Unknown") or "Unknown")
            road_name = str(row.get("Road_Name",       "Unknown") or "Unknown")

            # ── Street_Lit → dot size ─────────────────────────
            # Bigger dot = more dangerous lighting situation
            # Nighttime + unlit road = largest (most dangerous)
            # Nighttime + lit road   = medium
            # Daytime or unknown     = normal size
            street_lit = str(row.get("Street_Lit", "") or "").lower()
            dark_unlit = safe_int(row.get("Dark_Unlit", 0))
            dark_lit   = safe_int(row.get("Dark_Lit",   0))

            if dark_unlit == 1:
                dot_size  = 14   # Large — nighttime + confirmed unlit
                lit_text  = "&#10007; No streetlight (unlit road)"
                lit_color = "#CC0000"
            elif dark_lit == 1:
                dot_size  = 10   # Medium — nighttime but lit
                lit_text  = "&#10003; Streetlight present"
                lit_color = "#228B22"
            elif street_lit == "yes":
                dot_size  = 9    # Normal — daytime but lit road tagged
                lit_text  = "&#10003; Streetlight present"
                lit_color = "#228B22"
            elif street_lit == "no":
                dot_size  = 9    # Normal — daytime unlit road
                lit_text  = "&#10007; No streetlight (unlit road)"
                lit_color = "#CC0000"
            else:
                dot_size  = 7    # Default — lighting unknown
                lit_text  = "&#9898; Unknown (not tagged in OSM)"
                lit_color = "#888888"

            # Full weather details for popup
            temp        = row.get("Temperature", None)
            precip      = row.get("Precipitation", None)
            windspeed   = row.get("Windspeed", None)
            visibility  = row.get("Visibility", None)
            is_wet      = str(row.get("is_wet", "")).lower()

            temp_d    = f"{temp:.1f}°C"         if pd.notna(temp)       else "N/A"
            precip_d  = f"{precip:.1f} mm"      if pd.notna(precip)     else "N/A"
            wind_d    = f"{windspeed:.1f} km/h" if pd.notna(windspeed)  else "N/A"
            vis_d     = f"{int(visibility):,} m" if pd.notna(visibility) else "N/A"
            wet_d     = "Yes" if is_wet == "true" else \
                        "No"  if is_wet == "false" else "Unknown"

            aadt_d = f"{int(aadt):,} vehicles/day" if pd.notna(aadt) else "N/A"
            curv_d = f"{curvature:.3f}" if pd.notna(curvature) else "N/A"
            src_t  = {
                "station":            "&#10003; Measured station",
                "road_type_estimate": "&#9888; Road type estimate",
                "no_match":           "&#10007; No match"
            }.get(aadt_src, aadt_src)

            # Land use fields for popup
            zone_cat   = str(row.get("Zone_Category",  "Unknown") or "Unknown")
            zone_typ   = str(row.get("Zone_Type",       "Unknown") or "Unknown")
            dist_sch   = row.get("Dist_To_School",  None)
            near_sch   = safe_int(row.get("Near_School",    0))
            dist_bus   = row.get("Dist_To_Bus_Stop", None)
            near_bus   = safe_int(row.get("Near_Bus_Stop",  0))

            dist_sch_d = f"{dist_sch:.0f}m" if pd.notna(dist_sch) else "N/A"
            dist_bus_d = f"{dist_bus:.0f}m" if pd.notna(dist_bus) else "N/A"
            near_sch_d = "&#10003; Yes" if near_sch else "No"
            near_bus_d = "&#10003; Yes" if near_bus else "No"

            # Zone category color badge
            zone_colors = {
                "Residential":  "#228B22",
                "Commercial":   "#FF6600",
                "Industrial":   "#808080",
                "Mixed Use":    "#9400D3",
                "Civic / Public": "#2E75B6",
            }
            zone_color = zone_colors.get(zone_cat, "#888888")

            popup_html = f"""
            <div style="font-family:Arial;font-size:12px;
                        min-width:270px;line-height:1.8;">
              <div style="background:#1a1a2e;color:white;
                          padding:7px 12px;margin:-14px -20px 10px -20px;
                          border-radius:6px 6px 0 0;
                          font-weight:bold;font-size:13px;">
                Crash Details
              </div>
              <table style="width:100%;border-collapse:collapse;">
                <tr style="background:#f5f5f5;">
                  <td style="padding:3px 8px;color:#555;width:42%">Severity</td>
                  <td style="padding:3px 8px;">
                    <span style="background:{fill_color};color:white;
                                 padding:1px 7px;border-radius:3px;
                                 font-weight:bold;">
                      {sev} — {sev_label}
                    </span>
                  </td>
                </tr>
                <tr>
                  <td style="padding:3px 8px;color:#555;">Time of Day</td>
                  <td style="padding:3px 8px;">{time_text}</td>
                </tr>
                <tr style="background:#f5f5f5;">
                  <td style="padding:3px 8px;color:#555;">Road Name</td>
                  <td style="padding:3px 8px;font-weight:bold;">{road_name}</td>
                </tr>
                <tr>
                  <td style="padding:3px 8px;color:#555;">Road Type</td>
                  <td style="padding:3px 8px;">{road}</td>
                </tr>
                <tr style="background:#f5f5f5;">
                  <td style="padding:3px 8px;color:#555;">Speed Limit</td>
                  <td style="padding:3px 8px;">{speed}</td>
                </tr>
                <tr>
                  <td style="padding:3px 8px;color:#555;">Curvature</td>
                  <td style="padding:3px 8px;">{curv_d}
                    <span style="color:#888;font-size:10px;">
                      (1.0=straight)</span></td>
                </tr>
                <tr style="background:#f5f5f5;">
                  <td style="padding:3px 8px;color:#555;">Hour of Crash</td>
                  <td style="padding:3px 8px;">{hour}:00</td>
                </tr>
                <tr>
                  <td style="padding:3px 8px;color:#555;">AADT</td>
                  <td style="padding:3px 8px;">{aadt_d}</td>
                </tr>
                <tr style="background:#f5f5f5;">
                  <td style="padding:3px 8px;color:#555;">AADT Source</td>
                  <td style="padding:3px 8px;">{src_t}</td>
                </tr>
              </table>
              <div style="background:#f0fff0;border-left:3px solid #228B22;
                          padding:6px 10px;margin-top:8px;border-radius:3px;">
                <b style="font-size:12px;">&#127963; Land Use</b>
                <table style="width:100%;border-collapse:collapse;
                               margin-top:4px;">
                  <tr>
                    <td style="padding:2px 6px;color:#555;width:45%">
                      Zone</td>
                    <td style="padding:2px 6px;">
                      <span style="background:{zone_color};color:white;
                                   padding:1px 6px;border-radius:3px;
                                   font-size:11px;">
                        {zone_cat}
                      </span>
                      <span style="color:#888;font-size:10px;">
                        &nbsp;{zone_typ}
                      </span>
                    </td>
                  </tr>
                  <tr style="background:rgba(0,0,0,0.03);">
                    <td style="padding:2px 6px;color:#555;">
                      Nearest School</td>
                    <td style="padding:2px 6px;">{dist_sch_d}
                      &nbsp;<span style="color:{'#228B22' if near_sch
                        else '#888'};font-size:10px;">
                        {near_sch_d} (within 300m)
                      </span>
                    </td>
                  </tr>
                  <tr>
                    <td style="padding:2px 6px;color:#555;">
                      Nearest Bus Stop</td>
                    <td style="padding:2px 6px;">{dist_bus_d}
                      &nbsp;<span style="color:{'#2E75B6' if near_bus
                        else '#888'};font-size:10px;">
                        {near_bus_d} (within 150m)
                      </span>
                    </td>
                  </tr>
                </table>
              </div>
              <div style="background:#fffbe6;border-left:3px solid #FFA500;
                          padding:6px 10px;margin-top:8px;border-radius:3px;">
                <b style="font-size:12px;">&#128294; Street Lighting</b>
                <table style="width:100%;border-collapse:collapse;
                               margin-top:4px;">
                  <tr>
                    <td style="padding:2px 6px;color:#555;width:45%">
                      Streetlight</td>
                    <td style="padding:2px 6px;font-weight:bold;
                               color:{lit_color};">
                      {lit_text}
                    </td>
                  </tr>
                  <tr style="background:rgba(0,0,0,0.03);">
                    <td style="padding:2px 6px;color:#555;">
                      Time of Day</td>
                    <td style="padding:2px 6px;">{time_text}</td>
                  </tr>
                  <tr>
                    <td style="padding:2px 6px;color:#555;">
                      Dark + Unlit</td>
                    <td style="padding:2px 6px;font-weight:bold;
                               color:{'#CC0000' if dark_unlit else '#555'};">
                      {'YES — highest lighting risk' if dark_unlit
                       else 'No'}
                    </td>
                  </tr>
                </table>
                <div style="font-size:10px;color:#888;margin-top:4px;">
                  Dot size on map reflects lighting risk.
                  Larger dot = nighttime + unlit road.
                  Source: OpenStreetMap lit tag
                  (partial coverage — major roads only).
                </div>
              </div>
              <div style="background:#e8f4fd;border-left:3px solid #1E90FF;
                          padding:6px 10px;margin-top:8px;border-radius:3px;">
                <b style="font-size:12px;">&#127783; Weather at Crash Time</b>
                <table style="width:100%;border-collapse:collapse;
                               margin-top:4px;">
                  <tr>
                    <td style="padding:2px 6px;color:#555;width:45%">
                      Condition</td>
                    <td style="padding:2px 6px;">
                      <span style="background:{border_color};color:white;
                                   padding:1px 6px;border-radius:3px;
                                   font-size:11px;">
                        {cond_text}
                      </span>
                    </td>
                  </tr>
                  <tr style="background:rgba(0,0,0,0.03);">
                    <td style="padding:2px 6px;color:#555;">Wet Road</td>
                    <td style="padding:2px 6px;">{wet_d}</td>
                  </tr>
                  <tr>
                    <td style="padding:2px 6px;color:#555;">Temperature</td>
                    <td style="padding:2px 6px;">{temp_d}</td>
                  </tr>
                  <tr style="background:rgba(0,0,0,0.03);">
                    <td style="padding:2px 6px;color:#555;">Precipitation</td>
                    <td style="padding:2px 6px;">{precip_d}</td>
                  </tr>
                  <tr>
                    <td style="padding:2px 6px;color:#555;">Wind Speed</td>
                    <td style="padding:2px 6px;">{wind_d}</td>
                  </tr>
                  <tr style="background:rgba(0,0,0,0.03);">
                    <td style="padding:2px 6px;color:#555;">Visibility</td>
                    <td style="padding:2px 6px;">{vis_d}</td>
                  </tr>
                </table>
              </div>
            </div>
            """

            icon = make_icon(shape, fill_color, border_color, dot_size)
            folium.Marker(
                location=[lat, lon],
                icon=icon,
                popup=folium.Popup(popup_html, max_width=340),
                tooltip=(
                    f"{'&#9650; Night' if is_dark else '&#9679; Day'} | "
                    f"Sev {sev}: {sev_label} | "
                    f"{cond_text} | "
                    f"{'&#128294; Unlit' if dark_unlit else '&#128161; Lit' if street_lit == 'yes' else '&#9898; Lighting unknown'}"
                )
            ).add_to(sev_layers[sev])

        except Exception:
            continue

    for layer in sev_layers.values():
        layer.add_to(m)

    # Add land use layers — zone colors, schools, bus stops
    m = add_land_use_layers(m, df)

    # Layer control — severity toggles + land use layers
    folium.LayerControl(
        position="bottomleft",
        collapsed=False
    ).add_to(m)

    # Force layer control to bottom left so it does not
    # overlap the legend on the right side
    m.get_root().html.add_child(folium.Element("""
    <script>
    document.addEventListener("DOMContentLoaded", function() {
        var checkControl = setInterval(function() {
            var ctrl = document.querySelector(
                ".leaflet-bottom.leaflet-right "
                + ".leaflet-control-layers"
            );
            if (ctrl) {
                ctrl.style.cssText = (
                    "position:fixed !important;"
                    + "bottom:50px !important;"
                    + "left:15px !important;"
                    + "right:auto !important;"
                    + "max-height:280px;"
                    + "overflow-y:auto;"
                    + "font-size:12px;"
                    + "font-family:Arial;"
                    + "z-index:1000;"
                );
                clearInterval(checkControl);
            }
        }, 100);
    });
    </script>
    """))

    # Legend
    legend_html = """
    <div style="position:fixed;top:15px;right:15px;z-index:1000;
                background:white;padding:16px 20px;border-radius:10px;
                box-shadow:0 4px 15px rgba(0,0,0,0.25);max-width:268px;
                font-size:12px;line-height:1.8;font-family:Arial;">
      <h3 style="font-size:14px;font-weight:bold;color:#1a1a2e;
                 margin:0 0 12px;padding-bottom:6px;
                 border-bottom:3px solid #2E75B6;">
        &#128205; Map Legend
      </h3>
      <h4 style="font-size:11px;font-weight:bold;color:#2E75B6;
                 margin:8px 0 5px;text-transform:uppercase;">
        Shape = Time of Day
      </h4>
      <div style="display:flex;align-items:center;margin:4px 0;">
        <svg width="20" height="20" style="margin-right:9px;flex-shrink:0;">
          <polygon points="10,2 18,18 2,18"
            fill="#aaa" stroke="#333" stroke-width="2.5"
            stroke-linejoin="round"/>
        </svg>
        <span>Triangle &#9650; = Nighttime (8pm&ndash;6am)</span>
      </div>
      <div style="display:flex;align-items:center;margin:4px 0;">
        <svg width="20" height="20" style="margin-right:9px;flex-shrink:0;">
          <circle cx="10" cy="10" r="8"
            fill="#aaa" stroke="#333" stroke-width="2.5"/>
        </svg>
        <span>Circle &#9679; = Daytime (6am&ndash;8pm)</span>
      </div>

      <h4 style="font-size:11px;font-weight:bold;color:#FFA500;
                 margin:10px 0 5px;text-transform:uppercase;">
        Dot Size = Street Lighting Risk
      </h4>
      <div style="font-size:11px;color:#555;margin-bottom:5px;">
        Larger dot = more dangerous lighting situation
      </div>
      <div style="display:flex;align-items:center;margin:4px 0;">
        <svg width="20" height="20" style="margin-right:9px;flex-shrink:0;">
          <circle cx="10" cy="10" r="9"
            fill="#CC0000" stroke="#333" stroke-width="2"/>
        </svg>
        <span><b>Large (14px)</b> = Night + unlit road</span>
      </div>
      <div style="display:flex;align-items:center;margin:4px 0;">
        <svg width="18" height="18" style="margin-right:9px;flex-shrink:0;">
          <circle cx="9" cy="9" r="7"
            fill="#FF6600" stroke="#333" stroke-width="2"/>
        </svg>
        <span><b>Medium (10px)</b> = Night + lit road</span>
      </div>
      <div style="display:flex;align-items:center;margin:4px 0;">
        <svg width="14" height="14" style="margin-right:9px;flex-shrink:0;">
          <circle cx="7" cy="7" r="5"
            fill="#aaa" stroke="#333" stroke-width="1.5"/>
        </svg>
        <span><b>Small (7px)</b> = Daytime or unknown</span>
      </div>
      <h4 style="font-size:11px;font-weight:bold;color:#2E75B6;
                 margin:10px 0 5px;text-transform:uppercase;">
        Border Color = Weather Condition
      </h4>
      <div style="font-size:11px;color:#555;margin-bottom:6px;">
        Unknown weather crashes are hidden from the map.
      </div>
      <div style="display:flex;align-items:center;margin:3px 0;">
        <svg width="20" height="20" style="margin-right:9px;flex-shrink:0;">
          <circle cx="10" cy="10" r="8"
            fill="white" stroke="#87CEEB" stroke-width="3"/>
        </svg>
        <span>Sky blue = Clear</span>
      </div>
      <div style="display:flex;align-items:center;margin:3px 0;">
        <svg width="20" height="20" style="margin-right:9px;flex-shrink:0;">
          <circle cx="10" cy="10" r="8"
            fill="white" stroke="#B0C4DE" stroke-width="3"/>
        </svg>
        <span>Steel blue = Partly Cloudy</span>
      </div>
      <div style="display:flex;align-items:center;margin:3px 0;">
        <svg width="20" height="20" style="margin-right:9px;flex-shrink:0;">
          <circle cx="10" cy="10" r="8"
            fill="white" stroke="#4682B4" stroke-width="3"/>
        </svg>
        <span>Medium blue = Drizzle</span>
      </div>
      <div style="display:flex;align-items:center;margin:3px 0;">
        <svg width="20" height="20" style="margin-right:9px;flex-shrink:0;">
          <circle cx="10" cy="10" r="8"
            fill="white" stroke="#0000CD" stroke-width="3"/>
        </svg>
        <span>Dark blue = Rain</span>
      </div>
      <div style="display:flex;align-items:center;margin:3px 0;">
        <svg width="20" height="20" style="margin-right:9px;flex-shrink:0;">
          <circle cx="10" cy="10" r="8"
            fill="white" stroke="#1E90FF" stroke-width="3"/>
        </svg>
        <span>Dodger blue = Rain Showers</span>
      </div>
      <div style="display:flex;align-items:center;margin:3px 0;">
        <svg width="20" height="20" style="margin-right:9px;flex-shrink:0;">
          <circle cx="10" cy="10" r="8"
            fill="white" stroke="#8B008B" stroke-width="3"/>
        </svg>
        <span>Dark magenta = Thunderstorm</span>
      </div>
      <div style="display:flex;align-items:center;margin:3px 0;">
        <svg width="20" height="20" style="margin-right:9px;flex-shrink:0;">
          <circle cx="10" cy="10" r="8"
            fill="white" stroke="#708090" stroke-width="3"/>
        </svg>
        <span>Slate gray = Foggy</span>
      </div>
      <div style="display:flex;align-items:center;margin:3px 0;">
        <svg width="20" height="20" style="margin-right:9px;flex-shrink:0;">
          <circle cx="10" cy="10" r="8"
            fill="white" stroke="#00CED1" stroke-width="3"/>
        </svg>
        <span>Turquoise = Snow</span>
      </div>
      <div style="display:flex;align-items:center;margin:3px 0;">
        <svg width="20" height="20" style="margin-right:9px;flex-shrink:0;">
          <circle cx="10" cy="10" r="8"
            fill="white" stroke="#A9A9A9" stroke-width="3"/>
        </svg>
        <span>Dark gray = Other</span>
      </div>
      <h4 style="font-size:11px;font-weight:bold;color:#2E75B6;
                 margin:10px 0 5px;text-transform:uppercase;">
        Fill Color = Crash Severity
      </h4>
      <div style="display:flex;align-items:center;margin:3px 0;">
        <div style="width:15px;height:15px;border-radius:50%;
          background:#CC0000;margin-right:9px;border:1px solid #999;
          flex-shrink:0;"></div>
        <span><b>4</b> &mdash; Killed (Fatal)</span>
      </div>
      <div style="display:flex;align-items:center;margin:3px 0;">
        <div style="width:15px;height:15px;border-radius:50%;
          background:#FF6600;margin-right:9px;border:1px solid #999;
          flex-shrink:0;"></div>
        <span><b>1</b> &mdash; Incapacitating Injury</span>
      </div>
      <div style="display:flex;align-items:center;margin:3px 0;">
        <div style="width:15px;height:15px;border-radius:50%;
          background:#FFD700;margin-right:9px;border:1px solid #999;
          flex-shrink:0;"></div>
        <span><b>2</b> &mdash; Non-Incapacitating</span>
      </div>
      <div style="display:flex;align-items:center;margin:3px 0;">
        <div style="width:15px;height:15px;border-radius:50%;
          background:#90EE90;margin-right:9px;border:1px solid #999;
          flex-shrink:0;"></div>
        <span><b>3</b> &mdash; Possible Injury</span>
      </div>
      <div style="display:flex;align-items:center;margin:3px 0;">
        <div style="width:15px;height:15px;border-radius:50%;
          background:#228B22;margin-right:9px;border:1px solid #999;
          flex-shrink:0;"></div>
        <span><b>5</b> &mdash; Not Injured (PDO)</span>
      </div>
      <div style="display:flex;align-items:center;margin:3px 0;">
        <div style="width:15px;height:15px;border-radius:50%;
          background:#808080;margin-right:9px;border:1px solid #999;
          flex-shrink:0;"></div>
        <span><b>0</b> &mdash; Unknown Severity</span>
      </div>
      <h4 style="font-size:11px;font-weight:bold;color:#C71585;
                 margin:10px 0 5px;text-transform:uppercase;">
        Hotspot Rings — Repeated Crash Locations
      </h4>
      <div style="font-size:11px;color:#555;margin-bottom:6px;
                  background:#fff0f5;border-left:3px solid #C71585;
                  padding:5px 8px;border-radius:3px;">
        Pink/magenta rings mark locations where MULTIPLE crashes
        happened over time. Larger ring = more crashes = bigger
        safety problem. This color does not appear anywhere else
        on the map so it is impossible to confuse with severity.
      </div>
      <div style="display:flex;align-items:center;margin:5px 0;">
        <div style="width:22px;height:22px;border-radius:50%;
          border:4px solid #FF00FF;background:rgba(255,0,255,0.18);
          margin-right:9px;flex-shrink:0;"></div>
        <span><b>Bright magenta</b> = 5+ crashes<br>
        <span style="font-size:10px;color:#888;">
          Major problem area — largest ring</span></span>
      </div>
      <div style="display:flex;align-items:center;margin:5px 0;">
        <div style="width:20px;height:20px;border-radius:50%;
          border:3px solid #C71585;background:rgba(199,21,133,0.15);
          margin-right:9px;flex-shrink:0;"></div>
        <span><b>Deep pink</b> = 3&ndash;4 crashes<br>
        <span style="font-size:10px;color:#888;">
          High frequency location</span></span>
      </div>
      <div style="display:flex;align-items:center;margin:5px 0;">
        <div style="width:18px;height:18px;border-radius:50%;
          border:3px solid #FF69B4;background:rgba(255,105,180,0.12);
          margin-right:9px;flex-shrink:0;"></div>
        <span><b>Hot pink</b> = 2 crashes<br>
        <span style="font-size:10px;color:#888;">
          Repeated location</span></span>
      </div>
      <div style="background:#e8f4fd;border-left:3px solid #2E75B6;
                  padding:6px 8px;border-radius:3px;margin-top:10px;
                  font-size:11px;color:#444;">
        &#127760; Use layer control (bottom left) to toggle
        individual severity levels on/off<br><br>
        <b>Land use layers (off by default):</b><br>
        &#127963; Zone Categories — colored circles by zone type<br>
        &#127979; Schools — school locations<br>
        &#128652; Bus Stops — CapMetro stop locations
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Stats panel — shows weather condition breakdown
    night = int(df["Is_Dark"].sum()) if "Is_Dark" in df.columns else 0
    weather_counts = df["Weather_Condition"].value_counts()
    weather_rows = ""
    for cond, cnt in weather_counts.items():
        if str(cond) in SKIP_WEATHER:
            continue
        col = WEATHER_COLORS.get(str(cond), "#A9A9A9")
        weather_rows += (
            f'<div style="display:flex;justify-content:space-between;gap:10px;">'
            f'<span style="color:#555;">'
            f'<span style="display:inline-block;width:10px;height:10px;'
            f'border-radius:50%;border:2px solid {col};margin-right:4px;'
            f'vertical-align:middle;"></span>{cond}</span>'
            f'<span style="font-weight:bold;">{cnt}</span>'
            f'</div>'
        )
    shown  = int(df["Weather_Condition"].isin(list(WEATHER_COLORS.keys())).sum())
    hidden = 500 - shown
    stats_html = f"""
    <div style="position:fixed;top:15px;left:15px;z-index:1000;
                background:white;padding:14px 18px;border-radius:10px;
                box-shadow:0 4px 15px rgba(0,0,0,0.25);
                font-size:12px;line-height:1.9;font-family:Arial;
                max-width:230px;">
      <h3 style="font-size:13px;font-weight:bold;color:#1a1a2e;
                 margin:0 0 8px;padding-bottom:5px;
                 border-bottom:3px solid #2E75B6;">
        &#128200; Dataset Summary
      </h3>
      <div style="display:flex;justify-content:space-between;gap:10px;">
        <span style="color:#555;">Crashes shown</span>
        <span style="font-weight:bold;">{shown}</span>
      </div>
      <div style="display:flex;justify-content:space-between;gap:10px;">
        <span style="color:#aaa;">Hidden (unknown weather)</span>
        <span style="font-weight:bold;color:#aaa;">{hidden}</span>
      </div>
      <div style="display:flex;justify-content:space-between;gap:10px;">
        <span style="color:#555;">&#9650; Nighttime</span>
        <span style="font-weight:bold;">{night}</span>
      </div>
      <div style="display:flex;justify-content:space-between;gap:10px;">
        <span style="color:#555;">&#9679; Daytime</span>
        <span style="font-weight:bold;">{500-night}</span>
      </div>
      <hr style="margin:5px 0;border:none;border-top:1px solid #eee;">
      <b style="font-size:11px;color:#2E75B6;">Weather at Crash Time:</b>
      <div style="margin-top:3px;">{weather_rows}</div>
      <hr style="margin:5px 0;border:none;border-top:1px solid #eee;">
      <div style="display:flex;justify-content:space-between;gap:10px;">
        <span><b style="color:#CC0000;">4</b> Killed</span>
        <span style="font-weight:bold;color:#CC0000;">4</span>
      </div>
      <div style="display:flex;justify-content:space-between;gap:10px;">
        <span><b style="color:#FF6600;">1</b> Incapacitating</span>
        <span style="font-weight:bold;color:#FF6600;">29</span>
      </div>
      <div style="display:flex;justify-content:space-between;gap:10px;">
        <span><b style="color:#ccaa00;">2</b> Non-Incapacitating</span>
        <span style="font-weight:bold;color:#ccaa00;">100</span>
      </div>
      <div style="display:flex;justify-content:space-between;gap:10px;">
        <span><b style="color:#3a7a3a;">3</b> Possible Injury</span>
        <span style="font-weight:bold;">103</span>
      </div>
      <div style="display:flex;justify-content:space-between;gap:10px;">
        <span><b style="color:#228B22;">5</b> Not Injured</span>
        <span style="font-weight:bold;color:#228B22;">192</span>
      </div>
      <div style="display:flex;justify-content:space-between;gap:10px;">
        <span><b style="color:#808080;">0</b> Unknown</span>
        <span style="font-weight:bold;color:#808080;">72</span>
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(stats_html))

    m.get_root().html.add_child(folium.Element("""
    <div style="position:fixed;bottom:0;left:0;right:0;z-index:999;
                background:rgba(26,26,46,0.92);color:white;
                text-align:center;padding:7px;font-size:11px;
                font-family:Arial;letter-spacing:0.4px;">
      Austin Crash Safety Prediction System &nbsp;|&nbsp;
      Map 1: Individual Crashes &nbsp;|&nbsp;
      &#9650; Triangle=Night &nbsp; &#9679; Circle=Day &nbsp;|&nbsp;
      Border=Road Condition &nbsp;|&nbsp; Color=Severity &nbsp;|&nbsp;
      <span style="color:#FF69B4;font-weight:bold;">
        Pink/Magenta rings = repeated crash hotspots
      </span> &nbsp;|&nbsp;
      Severity filter: bottom left
    </div>
    """))

    m.save("crash_individual_map.html")
    print("  Saved: crash_individual_map.html")


# ── Map 2: Hotspot Grid Map ───────────────────────────────────
def build_hotspot_map(df):
    print("\nBuilding Map 2 — Crash hotspot grid...")

    # FIX 4: Tighter zoom
    m = folium.Map(
        location=[30.2672, -97.7431],
        zoom_start=12,
        tiles="CartoDB positron"
    )

    df["lat_r"] = df["latitude"].round(3)
    df["lon_r"] = df["longitude"].round(3)

    grid = df.groupby(["lat_r", "lon_r"]).agg(
        crash_count  = ("crash_sev_id", "count"),
        avg_severity = ("crash_sev_id", lambda x: x.apply(safe_int).mean()),
        fatal_count  = ("crash_sev_id", lambda x: (x.apply(safe_int) == 4).sum()),
        severe_count = ("crash_sev_id", lambda x: x.apply(safe_int).isin([1, 4]).sum()),
        night_count  = ("Is_Dark",      lambda x: x.apply(safe_int).sum()),
        wet_count    = ("is_wet",       lambda x: (x.astype(str).str.lower() == "true").sum()),
        dark_unlit   = ("Dark_Unlit",   lambda x: x.apply(safe_int).sum()),
        near_school  = ("Near_School",  lambda x: x.apply(safe_int).sum()),
        near_bus     = ("Near_Bus_Stop",lambda x: x.apply(safe_int).sum()),
        zone_cat     = ("Zone_Category", safe_mode),
        road_name    = ("Road_Name",        safe_mode),
        road_type    = ("Road_Type_Label",  safe_mode),
        highway_type = ("Highway_Type",     safe_mode)
    ).reset_index()

    # FIX 2: Flag trail cells in grid
    grid["is_trail"] = grid.apply(
        lambda r: is_trail(r["road_name"], r["highway_type"]), axis=1
    )

    max_count = int(grid["crash_count"].max())
    colormap  = cm.linear.YlOrRd_09.scale(1, max_count)
    colormap.caption = "Number of Crashes at This Location"

    grid_layer = folium.FeatureGroup(name="Crash Frequency Grid")

    for _, cell in grid.iterrows():
        count   = int(cell["crash_count"])
        lat     = cell["lat_r"]
        lon     = cell["lon_r"]
        road_n  = str(cell["road_name"]  or "Unknown")
        road_t  = str(cell["road_type"]  or "Unknown")
        fatals  = int(cell["fatal_count"])
        severe  = int(cell["severe_count"])
        nights  = int(cell["night_count"])
        wets    = int(cell["wet_count"])
        dunlit  = int(cell["dark_unlit"])
        nsch    = int(cell["near_school"])
        nbus    = int(cell["near_bus"])
        zone_c  = str(cell.get("zone_cat", "Unknown") or "Unknown")
        avg_sev = round(float(cell["avg_severity"]), 2)
        trail   = bool(cell["is_trail"])

        color   = colormap(count)
        opacity = min(0.88, 0.25 + count * 0.12)
        weight  = 1.5 if count >= 2 else 0.5

        sub = df[(df["lat_r"] == lat) & (df["lon_r"] == lon)]
        sev_counts = sub["crash_sev_id"].apply(safe_int).value_counts().sort_index()
        sev_rows = ""
        for code, cnt in sev_counts.items():
            dot_col = SEV_COLORS.get(code, "#808080")
            lbl     = SEV_LABELS.get(code, "Unknown")
            sev_rows += (
                f'<tr><td style="padding:2px 6px;">'
                f'<span style="background:{dot_col};color:white;'
                f'padding:1px 5px;border-radius:2px;font-size:11px;">'
                f'{code}</span></td>'
                f'<td style="padding:2px 6px;">{lbl}</td>'
                f'<td style="padding:2px 6px;font-weight:bold;">{cnt}</td>'
                f'</tr>'
            )

        # FIX 2: Show trail warning in popup if detected
        trail_warning = ""
        if trail:
            trail_warning = (
                '<div style="background:#fff3cd;border-left:3px solid #FF6600;'
                'padding:5px 8px;margin-bottom:6px;border-radius:3px;'
                'font-size:11px;">'
                '&#9888; Road matched to a trail/path. '
                'Actual crash road may differ.</div>'
            )

        popup_html = f"""
        <div style="font-family:Arial;font-size:12px;
                    min-width:280px;line-height:1.8;">
          <div style="background:#1a1a2e;color:white;padding:7px 12px;
                      margin:-14px -20px 10px -20px;
                      border-radius:6px 6px 0 0;
                      font-weight:bold;font-size:14px;">
            &#128293; Crash Hotspot
          </div>
          {trail_warning}
          <div style="background:#fff3cd;border-left:4px solid #FF6600;
                      padding:6px 10px;margin-bottom:8px;border-radius:3px;">
            <b>{count} crash{'es' if count > 1 else ''}</b>
            within ~100m of this point
          </div>
          <table style="width:100%;border-collapse:collapse;">
            <tr style="background:#f5f5f5;">
              <td style="padding:3px 8px;color:#555;width:45%">Road Name</td>
              <td style="padding:3px 8px;font-weight:bold;">{road_n}</td>
            </tr>
            <tr>
              <td style="padding:3px 8px;color:#555;">Road Type</td>
              <td style="padding:3px 8px;">{road_t}</td>
            </tr>
            <tr style="background:#f5f5f5;">
              <td style="padding:3px 8px;color:#555;">Fatal Crashes</td>
              <td style="padding:3px 8px;color:#CC0000;font-weight:bold;">
                {fatals}</td>
            </tr>
            <tr>
              <td style="padding:3px 8px;color:#555;">Severe Crashes</td>
              <td style="padding:3px 8px;color:#FF6600;font-weight:bold;">
                {severe}</td>
            </tr>
            <tr style="background:#f5f5f5;">
              <td style="padding:3px 8px;color:#555;">Nighttime Crashes</td>
              <td style="padding:3px 8px;">&#9650; {nights}</td>
            </tr>
            <tr>
              <td style="padding:3px 8px;color:#555;">Wet Road Crashes</td>
              <td style="padding:3px 8px;">&#128167; {wets}</td>
            </tr>
            <tr style="background:#fff5f5;">
              <td style="padding:3px 8px;color:#CC0000;font-weight:bold;">
                Dark + Unlit Crashes</td>
              <td style="padding:3px 8px;color:#CC0000;font-weight:bold;">
                &#128294; {dunlit}
                {'&nbsp;<span style="font-size:10px;">(adding a streetlight may help)</span>'
                 if dunlit > 0 else ''}
              </td>
            </tr>
            <tr style="background:#f5f5f5;">
              <td style="padding:3px 8px;color:#555;">Avg Severity</td>
              <td style="padding:3px 8px;">{avg_sev}</td>
            </tr>
            <tr>
              <td style="padding:3px 8px;color:#555;">Zone</td>
              <td style="padding:3px 8px;">{zone_c}</td>
            </tr>
            <tr style="background:#f5f5f5;">
              <td style="padding:3px 8px;color:#555;">Near School</td>
              <td style="padding:3px 8px;color:{'#228B22' if nsch > 0 else '#555'};">
                {'&#10003; ' + str(nsch) + ' crash(es) within 300m'
                 if nsch > 0 else 'No'}
              </td>
            </tr>
            <tr>
              <td style="padding:3px 8px;color:#555;">Near Bus Stop</td>
              <td style="padding:3px 8px;color:{'#2E75B6' if nbus > 0 else '#555'};">
                {'&#10003; ' + str(nbus) + ' crash(es) within 150m'
                 if nbus > 0 else 'No'}
              </td>
            </tr>
          </table>
          <br>
          <b style="font-size:12px;">Severity Breakdown:</b>
          <table style="width:100%;border-collapse:collapse;margin-top:5px;">
            <tr style="background:#f5f5f5;">
              <th style="padding:3px 6px;text-align:left;font-size:11px;">
                Code</th>
              <th style="padding:3px 6px;text-align:left;font-size:11px;">
                Label</th>
              <th style="padding:3px 6px;text-align:left;font-size:11px;">
                Count</th>
            </tr>
            {sev_rows}
          </table>
        </div>
        """

        folium.Rectangle(
            bounds=[
                [lat - 0.0005, lon - 0.0005],
                [lat + 0.0005, lon + 0.0005]
            ],
            color=color, fill=True, fill_color=color,
            fill_opacity=opacity, weight=weight,
            popup=folium.Popup(popup_html, max_width=330),
            tooltip=(
                f"{count} crash{'es' if count > 1 else ''} | "
                f"{road_n}"
                f"{' ⚠ Trail' if trail else ''}"
            )
        ).add_to(grid_layer)

    grid_layer.add_to(m)
    colormap.add_to(m)

    # FIX 2: Top 10 excludes trail cells
    grid_road = grid[~grid["is_trail"]]
    top10 = grid_road.nlargest(10, "crash_count")[
        ["lat_r", "lon_r", "crash_count", "road_name",
         "fatal_count", "severe_count", "night_count", "wet_count"]
    ].reset_index(drop=True)

    rows_html = ""
    for rank, row in top10.iterrows():
        bg    = "#fff3cd" if rank == 0 else \
                "#ffe5cc" if rank == 1 else \
                "#fff8e1" if rank < 5  else "white"
        medal = "&#127947;" if rank == 0 else \
                "&#129352;" if rank == 1 else \
                "&#129353;" if rank == 2 else f"#{rank+1}"
        rows_html += (
            f'<tr style="background:{bg};">'
            f'<td style="padding:3px 5px;text-align:center;">{medal}</td>'
            f'<td style="padding:3px 5px;font-weight:bold;color:#CC0000;">'
            f'{int(row["crash_count"])}</td>'
            f'<td style="padding:3px 5px;">'
            f'{str(row["road_name"])[:22]}</td>'
            f'<td style="padding:3px 5px;color:#CC0000;">'
            f'{int(row["fatal_count"])}</td>'
            f'<td style="padding:3px 5px;color:#555;">'
            f'&#9650;{int(row["night_count"])} '
            f'&#128167;{int(row["wet_count"])}</td>'
            f'</tr>'
        )

    stats_html = f"""
    <div style="position:fixed;top:15px;left:15px;z-index:1000;
                background:white;padding:14px 18px;border-radius:10px;
                box-shadow:0 4px 15px rgba(0,0,0,0.25);
                font-size:12px;font-family:Arial;max-width:375px;">
      <h3 style="font-size:13px;font-weight:bold;color:#1a1a2e;
                 margin:0 0 10px;padding-bottom:5px;
                 border-bottom:3px solid #CC0000;">
        &#128293; Top 10 Crash Hotspot Locations
        <span style="font-size:10px;color:#888;font-weight:normal;">
          (trails excluded)
        </span>
      </h3>
      <table style="width:100%;border-collapse:collapse;font-size:11px;">
        <tr style="background:#1a1a2e;color:white;">
          <th style="padding:4px 5px;">Rank</th>
          <th style="padding:4px 5px;">Crashes</th>
          <th style="padding:4px 5px;">Road Name</th>
          <th style="padding:4px 5px;">Fatal</th>
          <th style="padding:4px 5px;">Night/Wet</th>
        </tr>
        {rows_html}
      </table>
      <div style="font-size:10px;color:#888;margin-top:8px;">
        &#9650; = night crashes &nbsp;
        &#128167; = wet road crashes &nbsp;
        &#9888; = trail/path (click cell for warning)
        &bull; ~100m grid cells &bull; Click cell for details
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(stats_html))

    legend_html = """
    <div style="position:fixed;top:15px;right:15px;z-index:1000;
                background:white;padding:16px 20px;border-radius:10px;
                box-shadow:0 4px 15px rgba(0,0,0,0.25);max-width:245px;
                font-size:12px;font-family:Arial;line-height:1.8;">
      <h3 style="font-size:14px;font-weight:bold;color:#1a1a2e;
                 margin:0 0 10px;padding-bottom:6px;
                 border-bottom:3px solid #CC0000;">
        &#128205; Grid Legend
      </h3>
      <h4 style="font-size:11px;color:#CC0000;font-weight:bold;
                 margin:6px 0;text-transform:uppercase;">
        Cell Color = Crash Count
      </h4>
      <div style="display:flex;align-items:center;margin:4px 0;">
        <div style="width:22px;height:14px;background:#ffffb2;
                    margin-right:8px;border:1px solid #ccc;flex-shrink:0;">
        </div><span>1 crash</span>
      </div>
      <div style="display:flex;align-items:center;margin:4px 0;">
        <div style="width:22px;height:14px;background:#fecc5c;
                    margin-right:8px;border:1px solid #ccc;flex-shrink:0;">
        </div><span>2 crashes</span>
      </div>
      <div style="display:flex;align-items:center;margin:4px 0;">
        <div style="width:22px;height:14px;background:#fd8d3c;
                    margin-right:8px;border:1px solid #ccc;flex-shrink:0;">
        </div><span>3&ndash;4 crashes</span>
      </div>
      <div style="display:flex;align-items:center;margin:4px 0;">
        <div style="width:22px;height:14px;background:#f03b20;
                    margin-right:8px;border:1px solid #ccc;flex-shrink:0;">
        </div><span>5 crashes</span>
      </div>
      <div style="display:flex;align-items:center;margin:4px 0;">
        <div style="width:22px;height:14px;background:#bd0026;
                    margin-right:8px;border:1px solid #ccc;flex-shrink:0;">
        </div><span>6+ crashes (worst hotspot)</span>
      </div>
      <hr style="margin:10px 0;border:none;border-top:1px solid #eee;">
      <h4 style="font-size:11px;color:#9400D3;font-weight:bold;
                 margin:6px 0;text-transform:uppercase;">
        Zone Category Colors
      </h4>
      <div style="display:flex;align-items:center;margin:3px 0;">
        <div style="width:14px;height:14px;border-radius:50%;
          background:#228B22;margin-right:8px;flex-shrink:0;"></div>
        <span style="font-size:11px;">Residential</span>
      </div>
      <div style="display:flex;align-items:center;margin:3px 0;">
        <div style="width:14px;height:14px;border-radius:50%;
          background:#FF6600;margin-right:8px;flex-shrink:0;"></div>
        <span style="font-size:11px;">Commercial</span>
      </div>
      <div style="display:flex;align-items:center;margin:3px 0;">
        <div style="width:14px;height:14px;border-radius:50%;
          background:#9400D3;margin-right:8px;flex-shrink:0;"></div>
        <span style="font-size:11px;">Mixed Use</span>
      </div>
      <div style="display:flex;align-items:center;margin:3px 0;">
        <div style="width:14px;height:14px;border-radius:50%;
          background:#2E75B6;margin-right:8px;flex-shrink:0;"></div>
        <span style="font-size:11px;">Civic / Public</span>
      </div>
      <div style="display:flex;align-items:center;margin:3px 0;">
        <div style="width:14px;height:14px;border-radius:50%;
          background:#808080;margin-right:8px;flex-shrink:0;"></div>
        <span style="font-size:11px;">Industrial</span>
      </div>
      <hr style="margin:10px 0;border:none;border-top:1px solid #eee;">
      <div style="font-size:11px;color:#555;">
        Each cell covers ~100m x 100m on the ground.
        Darker red = more crashes concentrated there.
        Click any cell for full breakdown.<br><br>
        <b>Toggle in layer control (bottom left):</b><br>
        &#127979; Schools &nbsp; &#128652; Bus Stops<br>
        &#127963; Zone Categories
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    m.get_root().html.add_child(folium.Element("""
    <div style="position:fixed;bottom:0;left:0;right:0;z-index:999;
                background:rgba(26,26,46,0.92);color:white;
                text-align:center;padding:7px;font-size:11px;
                font-family:Arial;letter-spacing:0.4px;">
      Austin Crash Safety Prediction System &nbsp;|&nbsp;
      Map 2: Crash Hotspot Frequency Grid &nbsp;|&nbsp;
      Darker red = more crashes &nbsp;|&nbsp;
      Top 10 ranking excludes trails &nbsp;|&nbsp;
      Click any cell for full breakdown
    </div>
    """))

    # Add school and bus stop icon layers
    m = add_land_use_layers(m, df)

    folium.LayerControl(position="bottomleft").add_to(m)

    # Force layer control away from the legend
    m.get_root().html.add_child(folium.Element("""
    <script>
    document.addEventListener("DOMContentLoaded", function() {
        var checkControl = setInterval(function() {
            var ctrl = document.querySelector(
                ".leaflet-bottom.leaflet-right "
                + ".leaflet-control-layers"
            );
            if (ctrl) {
                ctrl.style.cssText = (
                    "position:fixed !important;"
                    + "bottom:50px !important;"
                    + "left:15px !important;"
                    + "right:auto !important;"
                    + "max-height:200px;"
                    + "overflow-y:auto;"
                    + "font-size:12px;"
                    + "font-family:Arial;"
                    + "z-index:1000;"
                );
                clearInterval(checkControl);
            }
        }, 100);
    });
    </script>
    """))
    m.save("crash_hotspot_map.html")
    print("  Saved: crash_hotspot_map.html")


# ── Main ──────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Austin Crash Safety — 2 Map Visualization Suite")
    print("=" * 55)

    df = load_data()
    build_individual_map(df)
    build_hotspot_map(df)

    print("\n" + "=" * 55)
    print("  Done! Two HTML maps generated:")
    print()
    print("  crash_individual_map.html")
    print("    FIX 1: Bigger shapes (22px) thicker borders (3px)")
    print("    FIX 3: Layer control toggles severity levels")
    print("    FIX 4: Zoomed to Austin city center")
    print()
    print("  crash_hotspot_map.html")
    print("    FIX 2: Purple Heart Trail excluded from top 10")
    print("    FIX 2: Trail cells show warning in popup")
    print("    FIX 4: Zoomed to Austin city center")
    print("=" * 55)


if __name__ == "__main__":
    main()