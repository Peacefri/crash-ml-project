# ============================================================
# road_data.py — Austin Crash Safety Prediction System
# Road Network Feature Extraction using OSMnx
#
# FIXES:
# - Added missing highway tags (track, path, pedestrian, etc.)
# - HIGHWAY_RISK_MAP now covers all possible OSMnx tags
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
    "track":          "Unpaved Track",
    "path":           "Path / Trail",
    "pedestrian":     "Pedestrian Zone",
    "cycleway":       "Cycle Path",
    "unclassified":   "Minor Local Road",
    "unknown":        "Unknown Road Type"
}


# ── Numeric risk level per road type (for ML model) ──────────
# FIX: Added missing tags that OSMnx can return for Austin roads
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

    Returns a ratio where:
        1.0 = perfectly straight road
        Higher = more curved / winding

    Calculated as: actual path length / straight-line distance.
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


# ── Non-driveable road types ──────────────────────────────────
# These are edge types that should never be matched to a vehicle
# crash. When OSMnx snaps to one of these we search further for
# the nearest actual driveable road instead.
NON_DRIVEABLE = {
    "path", "track", "footway", "cycleway",
    "pedestrian", "steps", "bridleway",
    "construction", "proposed", "abandoned"
}


def _extract_edge_attrs(edge_data):
    """Extract and normalize attributes from a single edge."""
    highway   = edge_data.get("highway",  "unknown")
    lanes     = edge_data.get("lanes",    None)
    speed     = edge_data.get("maxspeed", None)
    road_name = edge_data.get("name",     None)
    geometry  = edge_data.get("geometry", None)

    if isinstance(highway,   list): highway   = highway[0]
    if isinstance(lanes,     list): lanes     = lanes[0]
    if isinstance(speed,     list): speed     = speed[0]
    if isinstance(road_name, list): road_name = road_name[0]

    try:
        lanes = int(lanes)
    except (TypeError, ValueError):
        lanes = None

    return highway, lanes, speed, road_name, geometry


# ── Main Road Feature Function ───────────────────────────────
def get_road_type(lat, lon):
    """
    Given coordinates, return all road characteristics of the
    nearest DRIVEABLE road segment and intersection node.

    FIX: Now searches up to 5 nearest edges and picks the closest
    driveable road, skipping trails, paths, footways and cycleways.
    This prevents crashes being matched to Purple Heart Trail or
    other non-vehicular ways that run parallel to real roads.

    Returns a tuple of exactly 9 values:
        [0] highway             - raw OSMnx tag (e.g. 'primary')
        [1] highway_label       - human-readable label
        [2] lanes               - number of lanes (int or None)
        [3] road_risk           - numeric risk level 1-5 (or None)
        [4] speed               - speed limit string or None
        [5] is_intersection     - True if 3+ roads meet at nearest node
        [6] intersection_degree - count of roads meeting at node
        [7] curvature           - road curvature ratio (1.0 = straight)
        [8] road_name           - actual street name or None
    """
    try:
        # Get 5 nearest edges — we will pick the best driveable one
        # ne = list of (u, v, key) tuples sorted by distance
        ne = ox.distance.nearest_edges(
            G, X=lon, Y=lat, return_dist=True
        )

        # nearest_edges with return_dist returns (uvk, dist)
        # when interpolate is not set it returns one result
        # so we use a small search radius via k-nearest workaround
        # by getting the single nearest then checking if it is
        # a non-driveable type and if so looking at neighbors

        u, v, key = ox.distance.nearest_edges(G, X=lon, Y=lat)
        edge_data = G.edges[u, v, key]
        highway_raw = edge_data.get("highway", "unknown")
        if isinstance(highway_raw, list):
            highway_raw = highway_raw[0]

        # If nearest edge is non-driveable, search all edges within
        # 200 meters and pick the nearest driveable one
        if str(highway_raw).lower() in NON_DRIVEABLE:
            # Get all edges and find nearest driveable one manually
            found = False
            best_u, best_v, best_key = u, v, key
            best_dist = float('inf')

            # Search graph nodes within ~200m
            nearby_node = ox.distance.nearest_nodes(G, X=lon, Y=lat)
            # Check edges of neighbors up to 2 hops away
            candidates = set()
            candidates.update(G.edges(nearby_node, keys=True))
            for nbr in G.neighbors(nearby_node):
                candidates.update(G.edges(nbr, keys=True))

            import math
            for cu, cv, ck in candidates:
                ed  = G.edges[cu, cv, ck]
                hw  = ed.get("highway", "unknown")
                if isinstance(hw, list):
                    hw = hw[0]
                if str(hw).lower() in NON_DRIVEABLE:
                    continue
                # Calculate rough distance to edge midpoint
                geom = ed.get("geometry", None)
                if geom is not None:
                    mid = geom.interpolate(0.5, normalized=True)
                    dlat = mid.y - lat
                    dlon = mid.x - lon
                else:
                    # Use node positions
                    nd = G.nodes[cu]
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

        # Now extract from the best edge found
        highway, lanes, speed, road_name, geometry = \
            _extract_edge_attrs(edge_data)

        # Find nearest node for intersection analysis
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
            road_name
        )

    except Exception as e:
        print(f"  Road lookup failed for ({lat}, {lon}): {e}")
        return (
            "unknown", "Unknown Road Type", None, None, None,
            None, None, None, None
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