import pandas as pd
import geopandas as gpd
import networkx as nx
from geopy.distance import geodesic
from shapely.geometry import Point
from tqdm import tqdm
import requests as r
if __name__ == "__main__":
    #
    route_link = "http://localhost:5001/route/v1/driving/"
    #
    gdf = gpd.read_file("../data/stores_location.geojson")
    coords = [(point.y, point.x) for point in gdf.geometry]
    dict_name = {i:row["store_id"] for (i,row) in gdf.iterrows()}
    n = len(coords)
    #
    G = nx.Graph()
    for i in tqdm(range(n)):
        for j in range(n):
            if i != j:
                traj = [coords[i], coords[j]]
                coords_string = ';'.join([f"{lat},{lon}" for lon, lat in traj])
                link = route_link + coords_string
                link += "?alternatives=true&steps=false&geometries=geojson&annotations=nodes&continue_straight=false"
                x = r.get(link)
                js = x.json()
                jx = js["routes"][0]
                distance_route = jx["distance"]
                duration_route = jx["duration"]
                G.add_edge(dict_name[i], dict_name[j], **{"duration":duration_route,"distance":distance_route})
    nx.write_gml(G, "../data/distance_stores.gml")