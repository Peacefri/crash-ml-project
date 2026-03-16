import osmnx as ox
import pandas as pd

print("Loading Austin road network...")
G = ox.graph_from_place("Austin, Texas, USA", network_type="drive")
print("Road network loaded.")


def get_road_type(lat, lon):

    try:
        u, v, key = ox.distance.nearest_edges(G, X=lon, Y=lat)
        edge_data = G.edges[u, v, key]

        highway = edge_data.get("highway", "unknown")
        lanes = edge_data.get("lanes", None)

        if isinstance(highway, list):
            highway = highway[0]

        if isinstance(lanes, list):
            lanes = lanes[0]

        try:
            lanes = int(lanes)
        except:
            lanes = None

        return highway, lanes

    except Exception:
        return "unknown", None


def process_road(row):
    highway, lanes = get_road_type(row["latitude"], row["longitude"])
    return pd.Series({"Road_Type": highway, "Lanes": lanes})