import osmnx as ox
import pandas as pd


def get_road_type(lat, lon):
    try:
        G = ox.graph_from_place('Austin, Texas', network_type='drive')
        u, v, key = ox.distance.nearest_edges(G, X=lon, Y=lat)
        edge_data = G.edges[u, v, key]
        highway = edge_data.get('highway', 'unknown')
        lanes = edge_data.get('lanes', 'unknown')
        return highway, lanes
        if isinstance(highway, list): highway = highway[0]
        return highway, lanes
    except Exception:
        return 'unknown', 'unknown'
# helper fuction to handle row
def process_road(row):
    highway, lanes = get_road_type(row['Latitude'], row['Longitude'])
    return pd.Series([highway, lanes])
