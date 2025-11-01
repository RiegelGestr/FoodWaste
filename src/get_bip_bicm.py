import numpy as np
from datetime import datetime
import pandas as pd
import networkx as nx
from bicm import BipartiteGraph
import geopandas as gpd
from shapely.geometry import Point
import ast
from infomap import Infomap
import community.community_louvain as louvain
import powerlaw
from collections import Counter, defaultdict
from sklearn.metrics import adjusted_rand_score
from sklearn.utils import shuffle
import json


def ari_pvalue(true_labels, pred_labels, n_perm=9999):
    observed_ari = adjusted_rand_score(true_labels, pred_labels)
    permuted_aris = []
    for _ in range(n_perm):
        shuffled_labels = shuffle(pred_labels)
        permuted_aris.append(adjusted_rand_score(true_labels, shuffled_labels))
    permuted_aris = np.array(permuted_aris)
    p_value = (np.sum(permuted_aris >= observed_ari) + 1) / (n_perm + 1)
    return observed_ari, p_value


def compute_rca(X, eps=1e-10):
    row_sums = np.sum(X, axis=1, keepdims=True)
    col_sums = np.sum(X, axis=0, keepdims=True)
    total = np.sum(X)    
    store_product_share = X / row_sums    
    global_product_share = (col_sums + eps) / total    
    rca = store_product_share / global_product_share
    return rca


def hex_to_rgba(hex_color, alpha=1.0):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return {'r': r, 'g': g, 'b': b, 'a': alpha}
                

if __name__ == "__main__":
    df = pd.read_csv("../data/offers_data.csv")
    products_df = df[["product_ean","description"]]
    products_df = products_df.drop_duplicates()
    products_df["len"] = products_df["product_ean"].astype(str).apply(len)
    products_df = products_df[products_df["len"] == 13]
    #
    g = df.groupby(by = ["store_id","product_ean"]).count().reset_index()
    g = g[["store_id","product_ean","offer_ean"]]
    g = g.rename(columns = {"offer_ean":"weight"})
    #
    B = nx.Graph()
    stores_ids = g["store_id"].unique()
    B.add_nodes_from(stores_ids, bipartite=0)
    product_eans = g["product_ean"].unique()
    B.add_nodes_from(product_eans, bipartite=1)
    for _, row in g.iterrows():
        B.add_edge(row["store_id"], row["product_ean"],weight = row["weight"])
    dict_store_id = {i:stores_ids[i] for i in range(len(stores_ids))}
    dict_product_id = {i:product_eans[i] for i in range(len(product_eans))}
    adj_matrix = nx.bipartite.biadjacency_matrix(B, row_order=stores_ids, column_order=product_eans,weight="weight")
    mcp = adj_matrix.toarray()
    rca = compute_rca(mcp)
    #
    binary_rca = np.where(rca > 1, 1, 0)
    myGraph = BipartiteGraph()
    myGraph.set_biadjacency_matrix(binary_rca)
    myGraph.compute_projection(rows=True, alpha=0.05, approx_method='poisson', threads_num=4, progress_bar=True, validation_method='fdr')
    myGraph.compute_projection(rows=False, alpha=0.05, approx_method='poisson', threads_num=4, progress_bar=True, validation_method='fdr')
    #
    G_store = nx.Graph()
    for (k,vs) in myGraph.get_rows_projection().items():
        k_name = dict_store_id[k]
        for v in vs:
            v_name = dict_store_id[v]
            G_store.add_edge(k_name,v_name)
    dict_node = {node:id for (id,node) in enumerate(G_store.nodes())}
    rv_dict_node = {v:k for (k,v) in dict_node.items()}
    infomap = Infomap("--two-level")
    for edge in G_store.edges():
        u, v = edge
        ui = dict_node[u]
        vi = dict_node[v]
        infomap.add_link(ui,vi)
    infomap.run()
    communities = {node.node_id: node.module_id for node in infomap.nodes}
    store_community = pd.DataFrame([{
            "store_id":rv_dict_node[node],
            "community":com,
        } for (node,com) in communities.items()])
    partition = louvain.best_partition(G_store)
    communities = defaultdict(set)
    for node, comm in partition.items():
        communities[comm].add(node)
    community_list = list(communities.values())
    modularity_score = nx.algorithms.community.modularity(G_store, community_list)
    print(f"Modularity score: {modularity_score}")
    store_community["louvain"] = store_community["store_id"].map(partition)
    #
    dict_color_gephi = {
        0:'#999999',#noise
        1:'#2A3ACB',#capital region
        2:'#53C653',#countryside
        3:'#CA0D26'#metropolitan
    }
    kommune_stores = gpd.read_file("../data/stores_location.geojson")
    kommune_stores.index = kommune_stores.store_id
    dict_store_kommune = kommune_stores["NAME_2"].to_dict()
    store_community["kommune"] = store_community["store_id"].map(dict_store_kommune)
    kdf = pd.merge(kommune_stores,store_community,how = "inner",left_on = "NAME_2",right_on = "kommune")
    kgdf = gpd.GeoDataFrame(kdf, geometry='geometry', crs="EPSG:4326")
    #
    Gnew = nx.Graph()
    for edge in G_store.edges():
        u, v = edge
        Gnew.add_edge(u,v)
    for _,row in kgdf.iterrows():
        node = row["store_id"]
        Gnew.nodes[node]["degree_urbanization"] = row["DEGURBA_L1"]
        Gnew.nodes[node]["infomap"] = row["community"]
        Gnew.nodes[node]["louvain"] = row["louvain"]
        Gnew.nodes[node]["log_popolation"] = np.log10(row["Tot_Pop"])
        Gnew.nodes[node]["region"] = row["NAME_1"]
        Gnew.nodes[node]["kommune"] = row["NAME_2"]
        #
        lat = row["geometry_y"].y
        lon = row["geometry_y"].x
        Gnew.nodes[node]["latitude"] = lat
        Gnew.nodes[node]["longitude"] = lon
        #
        Gnew.nodes[node]['viz'] = {'color': hex_to_rgba(dict_color_gephi[row["louvain"]])}
        #
    nx.write_gexf(Gnew,"../data/projection_stores.gexf")
    #
    labels_infomap = kgdf["community"].tolist()
    labels_louvain = kgdf["louvain"].tolist()
    ari_score,pval = ari_pvalue(labels_infomap, labels_louvain)
    print(f"ARI: {ari_score:.3f}, p-value: {pval:.4f}")
    #Product
    G_prod = nx.Graph()
    for (k,vs) in myGraph.get_cols_projection().items():
        k_name = dict_product_id[k]
        for v in vs:
            v_name = dict_product_id[v]
            G_prod.add_edge(k_name,v_name)
    dict_node = {node:id for (id,node) in enumerate(G_prod.nodes())}
    rv_dict_node = {v:k for (k,v) in dict_node.items()}
    infomap = Infomap("--two-level")
    for edge in G_prod.edges():
        u, v = edge
        ui = dict_node[u]
        vi = dict_node[v]
        infomap.add_link(ui,vi)
    infomap.run()
    communities = {node.node_id: node.module_id for node in infomap.nodes}
    product_community = pd.DataFrame([{
            "product_id":rv_dict_node[node],
            "community":com,
        } for (node,com) in communities.items()])
    partition = louvain.best_partition(G_prod)
    communities = defaultdict(set)
    for node, comm in partition.items():
        communities[comm].add(node)
    community_list = list(communities.values())
    modularity_score = nx.algorithms.community.modularity(G_prod, community_list)
    print(f"Modularity score: {modularity_score}")
    product_community["louvain"] = product_community["product_id"].map(partition)
    #
    products_df_community = pd.merge(products_df,product_community,how = "inner",left_on = "product_ean",right_on = "product_id")
    #
    labels_infomap = products_df_community["community"].tolist()
    labels_louvain = products_df_community["louvain"].tolist()
    ari_score,pval = ari_pvalue(labels_infomap, labels_louvain)
    print(f"ARI: {ari_score:.3f}, p-value: {pval:.4f}")
    category_colors = {
        "dressings": "#9e0142", 
        "patee": "#d53e4f", 
        "meat": "#f46d43", 
        "bread": "#fdae61",
        "ham": "#fee08b", 
        "milk": "#ffffbf", 
        "desserts snacks": "#e6f598", 
        "rye breads": "#abdda4",
        "light breads": "#66c2a5",    
        "pork": "#2ca25f",         
        "chicken": "#006d2c",        
        "ready to eat meals": "#1C8FA6", 
        "cheese": "#2166ac",     
        "yoghurt": "#084594",       
        "sausage": "#5e3c99",  
        "beverages": "#762a83"     
        }
    Gnew = nx.Graph()
    for edge in G_prod.edges():
        u, v = edge
        Gnew.add_edge(u,v)
    for _,row in products_df_community.iterrows():
        node = row["product_id"]
        Gnew.nodes[node]["community"] = row["louvain"]
        Gnew.nodes[node]["infomap"] = row["community"]
        Gnew.nodes[node]["product_category"] = row["categories"]
        #
        Gnew.nodes[node]["nutriscore"] = row["nutriscore"]
        Gnew.nodes[node]["envscore"] = row["env"]
        if row["categories"] in category_colors:
            Gnew.nodes[node]['viz'] = {'color': hex_to_rgba(category_colors[row["categories"]])}
        else:
            Gnew.nodes[node]['viz'] = {'color': hex_to_rgba("#999999")}
    nx.write_gexf(Gnew,"../data/projection_products.gexf")