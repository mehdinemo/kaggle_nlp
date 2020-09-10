import pandas as pd
import networkx as nx


def jaccard_sim(edges: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    nodes.columns = ['node', 'node_weight']

    edges = edges.merge(nodes, how='left', left_on='source', right_on='node')
    edges = edges.merge(nodes, how='left', left_on='target', right_on='node')

    edges['jaccard_sim'] = edges['weight'] / (edges['node_weight_x'] + edges['node_weight_y'] - edges['weight'])
    edges.drop(edges.columns.difference(['source', 'target', 'jaccard_sim']), axis=1, inplace=True)
    # data['jaccard_sim'] = data['jaccard_sim'].round(4)

    return edges


def sim_nodes_detector(data_sim: pd.DataFrame) -> pd.DataFrame:
    sim_nodes = data_sim[(data_sim['jaccard_sim'] == 1) & (data_sim['source'] != data_sim['target'])]
    if sim_nodes.empty:
        return data_sim, pd.DataFrame()
    # sim_nodes = sim_nodes[sim_nodes['source'] != sim_nodes['target']]
    sim_nodes = sim_nodes.groupby('source')['target'].apply(list)

    sim_nodes = pd.DataFrame(sim_nodes)
    slist = []
    sim_dic = {}
    for index, row in sim_nodes.iterrows():
        if not index in slist:
            for s in row['target']:
                sim_dic.update({s: index})

        slist.extend(row['target'])

    data_sim = data_sim[(~data_sim['source'].isin(slist)) & (~data_sim['target'].isin(slist))]
    sim_df = pd.DataFrame.from_dict(sim_dic, orient='index')
    sim_df.columns = ['id']
    return data_sim, sim_df


def adj_matrix(G: nx.Graph, weight='weight'):
    all_nodes = list(G.nodes)
    sim = nx.to_numpy_array(G, weight=weight)

    sim = pd.DataFrame(sim)
    sim.index = all_nodes
    sim.columns = all_nodes

    return sim


def change_nodes(nodes: pd.DataFrame) -> pd.DataFrame:
    new_nodes = pd.DataFrame()
    for index, row in nodes.iterrows():
        ids = row['messages_ids'].split(',')
        tmp = pd.DataFrame(ids)
        tmp['word'] = row['keyword']
        new_nodes = new_nodes.append(tmp)
    new_nodes.columns = ['id', 'weight']
    new_nodes['id'] = new_nodes['id'].astype('int64')
    new_nodes = new_nodes.groupby(['id'], as_index=False)['weight'].count()

    return new_nodes


def sim_messages(edges, nodes, sim_thresh):
    sim_nodes = edges[edges['jaccard_sim'] >= sim_thresh]
    if sim_nodes.empty:
        return []

    G = nx.from_pandas_edgelist(sim_nodes, source='source', target='target', edge_attr=True)
    node_dic = dict(zip(nodes['node'], nodes['node_weight']))
    nx.set_node_attributes(G, node_dic, 'label')
    cc = nx.connected_components(G)
    sim_list = []
    for c in cc:
        sub_nodes = nodes[nodes['node'].isin(c)]
        sub_nodes = sub_nodes.sort_values(['node_weight'], ascending=False)
        sim_list.extend(sub_nodes.iloc[1:]['node'])

    return sim_list
