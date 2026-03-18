# ============================================================
# road_data.py — Austin Crash Safety Prediction System
# Road Network Feature Extraction using OSMnx
# ============================================================

import osmnx as ox
import pandas as pd
import numpy as np
import os
import pickle

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
    "unclassified":   1,
    "unknown":        None
}


# ── Load Graph ───────────────────────────────────────────────
def load_graph():
    if os.path.exists(GRAPH_CACHE_FILE):
        print("Loading road network from local cache...")
        with open(GRAPH_CACHE_FILE, "rb") as f:
            G = pickle.load(f)
        print("Road network loaded from cache.")
    else:
        print("Downloading Austin road network (first time only)...")
        G = ox.graph_from_place("Austin, Texas, USA", network_type="drive")
        with open(GRAPH_CACHE_FILE, "wb") as f:
            pickle.dump(G, f)
        print("Road network downloaded and cached locally.")
    return G


# Load once when module is first imported
G = load_graph()


# ── Curvature Calculation ────────────────────────────────────
def calculate_curvature(geometry):
    """
    Measures how much a road segment deviates from a straight line.
    1.0 = perfectly straight. Higher = more curved/winding.
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
        if straight_dist == 0:
            return None
        return round(actual_dist / straight_dist, 4)
    except Exception:
        return None


# ── Main Road Feature Function ───────────────────────────────
def get_road_type(lat, lon):
    """
    Returns a tuple of EXACTLY 9 values:
        [0] highway             - raw OSMnx tag e.g. 'primary'
        [1] highway_label       - human-readable e.g. 'Major City Road'
        [2] lanes               - number of lanes (int or None)
        [3] road_risk           - numeric risk level 1-5 (or None)
        [4] speed               - speed limit string or None
        [5] is_intersection     - True if 3+ roads meet at nearest node
        [6] intersection_degree - count of roads meeting at nearest node
        [7] curvature           - road curvature ratio (1.0 = straight)
        [8] road_name           - actual street name or None
    """
    try:
        # Find nearest road segment
        u, v, key = ox.distance.nearest_edges(G, X=lon, Y=lat)
        edge_data = G.edges[u, v, key]

        # Find nearest node for intersection info
        nearest_node        = ox.distance.nearest_nodes(G, X=lon, Y=lat)
        intersection_degree = G.degree(nearest_node)
        is_intersection     = intersection_degree >= 3

        # Extract raw attributes
        highway   = edge_data.get("highway",  "unknown")
        lanes     = edge_data.get("lanes",    None)
        speed     = edge_data.get("maxspeed", None)
        road_name = edge_data.get("name",     None)
        geometry  = edge_data.get("geometry", None)

        # OSMnx sometimes returns lists — take first value
        if isinstance(highway,   list): highway   = highway[0]
        if isinstance(lanes,     list): lanes     = lanes[0]
        if isinstance(speed,     list): speed     = speed[0]
        if isinstance(road_name, list): road_name = road_name[0]

        # Convert lanes to int safely
        try:
            lanes = int(lanes)
        except (TypeError, ValueError):
            lanes = None

        # Derived features
        highway_label = HIGHWAY_LABELS.get(highway, "Other Road")
        road_risk     = HIGHWAY_RISK_MAP.get(highway, None)
        curvature     = calculate_curvature(geometry)

        return (
            highway,            # [0]
            highway_label,      # [1]
            lanes,              # [2]
            road_risk,          # [3]
            speed,              # [4]
            is_intersection,    # [5]
            intersection_degree,# [6]
            curvature,          # [7]
            road_name           # [8]
        )

    except Exception as e:
        print(f"  Road lookup failed for ({lat}, {lon}): {e}")
        return (
            "unknown",          # [0]
            "Unknown Road Type",# [1]
            None,               # [2]
            None,               # [3]
            None,               # [4]
            None,               # [5]
            None,               # [6]
            None,               # [7]
            None                # [8]
        )


# ── Row-wise Wrapper ─────────────────────────────────────────
def process_road(row):
    """Row-wise wrapper for use with df.apply()"""
    (highway, highway_label, lanes, road_risk, speed,
     is_intersection, intersection_degree,
     curvature, road_name) = get_road_type(
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
        "Road_Name":           road_name
    })
