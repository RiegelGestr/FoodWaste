import numpy as np
import networkx as nx
import pandas as pd
from collections import Counter, defaultdict
from tqdm import tqdm
from scipy.spatial.distance import cdist
import numpy as np
from skbio import DistanceMatrix
from skbio.stats.distance import mantel,permanova
import hdbscan

def get_cat_by_store():
    cat_by_store = defaultdict(dict)
    for _,row in df.iterrows():
        id = row['store_id']
        cat = row['categories']
        if cat in cat_by_store[id].keys(): cat_by_store[id][cat] += 1
        else: cat_by_store[id][cat] = 1
    all_cats = set()    
    for store, d in cat_by_store.items():
        for c in d.keys(): all_cats.add(c)
    list_cats = [(c,i) for i,c in enumerate(sorted(all_cats)) ]

    return cat_by_store, list_cats

def get_cat_vec():
    
    category_vectors = defaultdict(list)
    for store, d in cat_by_store.items():
        for c, i in list_cats:
            category_vectors[store].append(d.get(c,0))
    return category_vectors


if __name__ == "__main__":
    df = pd.read_csv("../Data/offers_data.csv",sep = ";")
    cat_by_store, list_cats = get_cat_by_store()
    category_vectors = get_cat_vec()
    len_cats = len(list(category_vectors.values())[0])
    mapping_region = {row['store_id']:row["Region"] for i,row in df.iterrows()}
    #
    stores = df["store_id"].tolist()
    G = nx.read_gml('../data/distance_stores.gml')
    nG = nx.Graph()
    for (u,v,d) in G.edges(data = True):
        if u not in stores or v not in stores: continue
        nG.add_edge(u,v,weight = d['distance'])
    mapping_id_label = {i:label for i,label in enumerate(nG.nodes())}
    #
    distances_df = pd.read_csv("data/stores/hdbscan_distances.csv",sep = ",")
    #
    max_distance = distances_df.values.max()
    min_distance = distances_df.values.min()
    n_bins = 1_000
    bins = np.linspace(500, max_distance, n_bins + 1)
    M = distances_df.values
    tmp = []
    pop = 0
    for bin in tqdm(bins):
        condition = (M > 0) & (M <= bin)
        G = nx.Graph()
        rows, cols = np.where(condition)
        G.add_edges_from(zip(rows, cols))
        if len(G) == 0:
            tmp.append({
            "distance":bin,
            "largest_cc":0,
            "second_largest_cc": 0,
            "number_of_stores": 0,
            "regions": set()
        })
        else:
            components = list(nx.connected_components(G))
            sorted_components = sorted(components, key=len, reverse=True)        
            largest_cc_size = len(sorted_components[0])
            second_largest_cc_size = len(sorted_components[1]) if len(sorted_components) > 1 else 0
            largest_cc = sorted_components[0]
            regions = len(set([mapping_region[mapping_id_label[i]] for i in largest_cc]))
            counter_regions = dict(Counter([mapping_region[mapping_id_label[i]] for i in largest_cc]))
            tmp.append({
                "distance": bin,
                "largest_cc": largest_cc_size,
                "second_largest_cc": second_largest_cc_size,
                "number_of_stores": len(G),
                "n_regions": regions,
                "regions":counter_regions,
            })
    giant_component_df = pd.DataFrame(tmp)
    #
    product_matrix = np.array([np.array(category_vectors[mapping_id_label[i]]) for i in range(len(mapping_id_label))])
    pm = product_matrix/product_matrix.sum(axis = 1,keepdims=True)
    bray_curtis_matrix_full = cdist(pm, pm, metric='braycurtis')
    bray_curtis_dm_skbio = DistanceMatrix(bray_curtis_matrix_full, ids=list(mapping_id_label.values()))
    geo_distance_dm_skbio = DistanceMatrix(distances_df.values, ids=distances_df.index)
    #
    max_distance = distances_df.values.max()
    min_distance = distances_df.values.min()
    n_bins = 1_000
    bins = np.linspace(500, max_distance, n_bins + 1)
    #
    mantel_r_values = []
    p_values = []
    mantel_r_distance = []
    p_values_distance = []

    #
    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(n_bins)]
    #
    for i in tqdm(range(n_bins)):
        lower_bound = bins[i]
        upper_bound = bins[i+1]
        binary_geo_dist_bin_array = ((distances_df > 0) & (distances_df <= upper_bound)).astype(int).values
        value_dist_bin_array = binary_geo_dist_bin_array*distances_df.values
        if binary_geo_dist_bin_array.sum() > 0:
            binary_geo_dm_skbio = DistanceMatrix(binary_geo_dist_bin_array, ids=distances_df.index)        
            coeff, p_value, _ = mantel(binary_geo_dm_skbio, bray_curtis_dm_skbio, permutations=999)
            mantel_r_values.append(coeff)
            p_values.append(p_value)
            binary_geo_dm_skbio = DistanceMatrix(value_dist_bin_array, ids=distances_df.index)        
            coeff, p_value, _ = mantel(binary_geo_dm_skbio, bray_curtis_dm_skbio, permutations=999)
            mantel_r_distance.append(coeff)
            p_values_distance.append(p_value)
        else:
            mantel_r_values.append(np.nan)
            p_values.append(np.nan)
            mantel_r_distance.append(np.nan)
            p_values_distance.append(np.nan)

    mantel_df = pd.DataFrame()
    mantel_df["p_value"] = p_values
    mantel_df["mantel_r"] = mantel_r_values
    mantel_df["bin_centers"] = bin_centers
    mantel_df["p_value_distance"] = p_values_distance
    mantel_df["mantel_distance"] = mantel_r_distance
    #
    giant_component_df.to_csv("../data/giant_component.csv",index = False)
    mantel_df.to_csv("../data/mantel_correlogram.csv",index = False)
    #
    matrix_distance = nx.to_scipy_sparse_array(nG)
    clustering = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size = 10)
    clustering.fit(matrix_distance)
    cluster_labels = clustering.labels_
    permanova_result = permanova(bray_curtis_dm_skbio, cluster_labels, permutations=999)
    print(permanova_result)
    #