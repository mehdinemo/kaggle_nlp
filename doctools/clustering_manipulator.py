import community
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from doctools import prepare_data as pr


def clustering_graph(G: nx.Graph, noise_deletion=True, eps=0.9, min_samples=5) -> pd.DataFrame:
    sim = pr.adj_matrix(G)
    dis = 1 - sim

    if noise_deletion:
        dbs = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(dis)

        # Remove Noises from Graph
        noise_nodes = np.where(dbs.labels_ == -1)[0]
        noise_nodes = dis.index[noise_nodes]
        G.remove_nodes_from(noise_nodes)

    partitions = community.best_partition(G)

    if noise_deletion:
        noise_dic = {k: -1 for k in noise_nodes}
        partitions.update(noise_dic)

    partitions = pd.DataFrame.from_dict(partitions, orient='index')
    partitions.columns = ['class']

    return partitions
