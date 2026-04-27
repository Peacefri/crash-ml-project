# ============================================================
# road_data.py — Austin Crash Safety Prediction System
# Road Network Feature Extraction using OSMnx
#
# Features extracted per crash location:
#   Highway_Type, Road_Type_Label, Road_Name
#   Num_Lanes, Speed_Limit, Road_Risk_Level
#   Is_Intersection, Intersection_Degree
#   Road_Curvature, Street_Lit
#
# FIXES:
#   - Added missing highway tags (track, path, pedestrian etc)
#   - HIGHWAY_RISK_MAP covers all possible OSMnx tags
#   - Non-driveable road filter (trails, paths, footways)
#   - Trail road name leak fix
#   - Street_Lit tag added (yes/no/None from OpenStreetMap)
#   - Curvature near-zero threshold fix (was == 0, now < 1e-10)
# ============================================================

import osmnx as ox
import pandas as pd
import numpy as np
import os
import pickle
import math

GRAPH_CACHE_FILE = "austin_road_network.pkl"


# ── Friendly road type labels for visualizations ─────────────
HIGHWAY_LABELS = {
    "motorway":       "Interstate / Freeway",
    "motorway_link":  "Freeway On/Off Ramp",
    "trunk":          "Major State Highway",
    "trunk_link":     "State Highway Ramp",
    "primary":        "Major City Road",
    "primary_link":   "Major City Road Ramp",
    "secondary":      "City Connector Road",
    "secondary_link": "City Connector Ramp",
    "tertiary":       "Neighborhood Connector",
    "tertiary_link":  "Neighborhood Connector Ramp",
    "residential":    "Residential Street",
    "living_street":  "Shared Living Street",
    "service":        "Service / Parking Road",
    "track":          "Unpaved Track",
    "path":           "Path / Trail",
    "pedestrian":     "Pedestrian Zone",
    "cycleway":       "Cycle Path",
    "unclassified":   "Minor Local Road",
    "unknown":        "Unknown Road Type"
}


# ── Numeric risk level per road type (for ML model) ──────────
HIGHWAY_RISK_MAP = {
    "motorway":       5,
    "motorway_link":  5,
    "trunk":          4,
    "trunk_link":     4,
    "primary":        3,
    "primary_link":   3,
    "secondary":      3,
    "secondary_link": 3,
    "tertiary":       2,
    "tertiary_link":  2,
    "residential":    1,
    "living_street":  1,
    "service":        1,
    "track":          1,
    "path":           1,
    "pedestrian":     1,
    "cycleway":       1,
    "unclassified":   1,
    "unknown":        None
}


# ── Non-driveable road types ──────────────────────────────────
# When OSMnx snaps to one of these we search further for the
# nearest actual driveable road instead.
NON_DRIVEABLE = {
    "path", "track", "footway", "cycleway",
    "pedestrian", "steps", "bridleway",
    "construction", "proposed", "abandoned"
}

# ── Trail name keywords ───────────────────────────────────────
# If the road name contains any of these words, clear it so
# the crash address is used instead.
TRAIL_NAME_KEYWORDS = [
    "trail", "trailhead", "greenway", "greenbelt",
    "hike", "bike path", "creek path", "nature path"
]


# ── Load Graph ───────────────────────────────────────────────
def load_graph():
    """
    Load Austin road network from local cache or download fresh.
    Caching avoids re-downloading every run (~30-60 seconds saved).
    """
    if os.path.exists(GRAPH_CACHE_FILE):
        print("Loading road network from local cache...")
        with open(GRAPH_CACHE_FILE, "rb") as f:
            G = pickle.load(f)
        print("Road network loaded from cache.")
    else:
        print("Downloading Austin road network (first time only)...")
        G = ox.graph_from_place(
            "Austin, Texas, USA",
            network_type="drive"
        )
        with open(GRAPH_CACHE_FILE, "wb") as f:
            pickle.dump(G, f)
        print("Road network downloaded and cached locally.")
    return G


# Load once when module is imported
G = load_graph()


# ── Curvature Calculation ────────────────────────────────────
def calculate_curvature(geometry):
    """
    Measures how much a road segment deviates from a straight line.
    Returns ratio where 1.0 = perfectly straight, higher = more curved.

    FIX: Uses < 1e-10 threshold instead of == 0 to avoid returning
    astronomically large ratios for near-zero straight distances.
    """
    try:
        if geometry is None or geometry.geom_type != "LineString":
            return None
        coords = list(geometry.coords)
        if len(coords) < 3:
            return 1.0
        start = np.array(coords[0])
        end   = np.array(coords[-1])
        straight_dist = np.linalg.norm(end - start)
        actual_dist = sum(
            np.linalg.norm(np.array(coords[i + 1]) - np.array(coords[i]))
            for i in range(len(coords) - 1)
        )
        # FIX: was `if straight_dist == 0` — now uses safe threshold
        if straight_dist < 1e-10:
            return None
        return round(actual_dist / straight_dist, 4)
    except Exception:
        return None


# ── Edge Attribute Extractor ─────────────────────────────────
def _extract_edge_attrs(edge_data):
    """
    Extract and normalize all attributes from a single OSMnx edge.
    Returns: highway, lanes, speed, road_name, geometry, lit
    """
    highway   = edge_data.get("highway",  "unknown")
    lanes     = edge_data.get("lanes",    None)
    speed     = edge_data.get("maxspeed", None)
    road_name = edge_data.get("name",     None)
    geometry  = edge_data.get("geometry", None)
    lit       = edge_data.get("lit",      None)

    # OSMnx sometimes returns lists — always take first value
    if isinstance(highway,   list): highway   = highway[0]
    if isinstance(lanes,     list): lanes     = lanes[0]
    if isinstance(speed,     list): speed     = speed[0]
    if isinstance(road_name, list): road_name = road_name[0]
    if isinstance(lit,       list): lit       = lit[0]

    try:
        lanes = int(lanes)
    except (TypeError, ValueError):
        lanes = None

    # Normalize lit to standard values
    if lit is not None:
        lit = str(lit).strip().lower()
        if lit not in ("yes", "no"):
            lit = None   # Ignore non-standard values

    return highway, lanes, speed, road_name, geometry, lit


# ── Main Road Feature Function ───────────────────────────────
def get_road_type(lat, lon):
    """
    Given coordinates, return all road characteristics of the
    nearest DRIVEABLE road segment and intersection node.

    Skips trails, paths, footways and cycleways.
    Clears road names that look like trail names.

    Returns a tuple of exactly 10 values:
        [0] highway             - raw OSMnx tag (e.g. 'primary')
        [1] highway_label       - human-readable label
        [2] lanes               - number of lanes (int or None)
        [3] road_risk           - numeric risk level 1-5 (or None)
        [4] speed               - speed limit string or None
        [5] is_intersection     - True if 3+ roads meet at nearest node
        [6] intersection_degree - count of roads meeting at node
        [7] curvature           - road curvature ratio (1.0=straight)
        [8] road_name           - actual street name or None
        [9] lit                 - 'yes', 'no', or None (OSMnx lit tag)
    """
    try:
        # Find nearest edge
        u, v, key = ox.distance.nearest_edges(G, X=lon, Y=lat)
        edge_data = G.edges[u, v, key]
        highway_raw = edge_data.get("highway", "unknown")
        if isinstance(highway_raw, list):
            highway_raw = highway_raw[0]

        # If nearest edge is non-driveable, find nearest driveable one
        if str(highway_raw).lower() in NON_DRIVEABLE:
            found     = False
            best_u, best_v, best_key = u, v, key
            best_dist = float('inf')

            nearby_node = ox.distance.nearest_nodes(G, X=lon, Y=lat)
            candidates  = set()
            candidates.update(G.edges(nearby_node, keys=True))
            for nbr in G.neighbors(nearby_node):
                candidates.update(G.edges(nbr, keys=True))

            for cu, cv, ck in candidates:
                ed = G.edges[cu, cv, ck]
                hw = ed.get("highway", "unknown")
                if isinstance(hw, list):
                    hw = hw[0]
                if str(hw).lower() in NON_DRIVEABLE:
                    continue
                geom = ed.get("geometry", None)
                if geom is not None:
                    mid  = geom.interpolate(0.5, normalized=True)
                    dlat = mid.y - lat
                    dlon = mid.x - lon
                else:
                    nd   = G.nodes[cu]
                    dlat = nd.get('y', lat) - lat
                    dlon = nd.get('x', lon) - lon
                dist = math.sqrt(dlat**2 + dlon**2)
                if dist < best_dist:
                    best_dist = dist
                    best_u, best_v, best_key = cu, cv, ck
                    found = True

            if found:
                u, v, key = best_u, best_v, best_key
                edge_data = G.edges[u, v, key]

        # Extract all attributes from chosen edge
        highway, lanes, speed, road_name, geometry, lit = \
            _extract_edge_attrs(edge_data)

        # Clear road name if it looks like a trail name
        if road_name:
            name_lower = str(road_name).lower()
            if any(kw in name_lower for kw in TRAIL_NAME_KEYWORDS):
                road_name = None

        # Nearest node for intersection analysis
        nearest_node        = ox.distance.nearest_nodes(G, X=lon, Y=lat)
        intersection_degree = G.degree(nearest_node)
        is_intersection     = intersection_degree >= 3

        highway_label = HIGHWAY_LABELS.get(highway, "Other Road")
        road_risk     = HIGHWAY_RISK_MAP.get(highway, None)
        curvature     = calculate_curvature(geometry)

        return (
            highway,
            highway_label,
            lanes,
            road_risk,
            speed,
            is_intersection,
            intersection_degree,
            curvature,
            road_name,
            lit
        )

    except Exception as e:
        print(f"  Road lookup failed for ({lat}, {lon}): {e}")
        return (
            "unknown", "Unknown Road Type", None, None, None,
            None, None, None, None, None
        )


# ── Row-wise Wrapper (unused — kept for Phase 2 df.apply use) ─
def process_road(row):
    """Row-wise wrapper for use with df.apply() in Phase 2"""
    (highway, highway_label, lanes, road_risk, speed,
     is_intersection, intersection_degree,
     curvature, road_name, lit) = get_road_type(
        row["latitude"], row["longitude"]
    )

    return pd.Series({
        "Highway_Type":        highway,
        "Road_Type_Label":     highway_label,
        "Num_Lanes":           lanes,
        "Road_Risk_Level":     road_risk,
        "Speed_Limit":         speed,
        "Is_Intersection":     is_intersection,
        "Intersection_Degree": intersection_degree,
        "Road_Curvature":      curvature,
        "Road_Name":           road_name,
        "Street_Lit":          lit
    })