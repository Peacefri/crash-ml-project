import osmnx as ox

def get_road_type(lat, lon):
    try:
        G = ox.graph_from_point((lat, lon), dist=100, network_type='drive')
        u, v, key = ox.distance.nearest_edges(G, X=lon, Y=lat)
        edge_data = G.edges[u, v, key]
        highway = edge_data.get('highway', 'unknown')
        lanes = edge_data.get('lanes', 'unknown')
        return highway, lanes
    except Exception as e:
        return 'unknown', 'unknown'