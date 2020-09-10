import networkx as nx
import pandas as pd
from datetime import datetime
from doctools import text_tools as tt
from doctools import prepare_data as pr
from doctools import clustering_manipulator as cm
from doctools import classification_tool as ct
import json


def _validate_input(data, data_columns):
    # بررسی صحت پارامتر های بدنه
    # شامل فیلدهای مورد نظر باشد
    if 'messages' not in data:
        return False, f"messages dos'nt exit in data!!"
    for idx, msg in enumerate(data["messages"]):
        for col in data_columns:
            if not col in msg:
                return False, f'record {idx} does not have {col}!'

    return True, None


def document_clustering(messages_df, prune_thresh, noise_deletion, eps, min_samples):
    edges, nodes = tt.create_graph(messages_df.copy())

    data_sim = pr.jaccard_sim(edges, nodes)
    data_sim, sim_df = pr.sim_nodes_detector(data_sim)

    # prune edges
    data_sim = data_sim[data_sim['jaccard_sim'] >= prune_thresh]

    if len(data_sim) < 1:
        return {'warning': 'empty Graph'}

    G = nx.from_pandas_edgelist(data_sim, source='source', target='target', edge_attr=True)

    partitions = cm.clustering_graph(G.copy(), noise_deletion=noise_deletion, eps=eps, min_samples=min_samples)

    # add similar messages
    sim_df = sim_df.merge(partitions, how='left', left_on='id', right_index=True)
    sim_df.drop(['id'], axis=1, inplace=True)
    partitions = partitions.append(sim_df)

    messages_df = messages_df.merge(partitions, how='inner', left_on='id', right_index=True)
    messages_df.fillna(-1, inplace=True)
    messages_df['class'] = messages_df['class'].astype(int)

    return messages_df


def similar_messages(messages_df, sim_thresh):
    edges, nodes = tt.create_graph(messages_df.drop(messages_df.columns.difference(['id', 'text']), axis=1).copy())
    data_sim = pr.jaccard_sim(edges, nodes)

    sim_list = pr.sim_messages(data_sim, nodes, sim_thresh)

    messages_df = messages_df[~messages_df['id'].isin(sim_list)]

    return messages_df


def document_classification(tagged_messages, untagged_messages, method='degree', sub_method='degree',
                            label_method='mean',
                            n_head_score=1):
    tagged_messages['tm'] = 1
    untagged_messages['tm'] = 0

    messages_df = pd.concat([tagged_messages, untagged_messages], ignore_index=True, sort=True)
    messages_df.reset_index(inplace=True)

    tagged_messages = tagged_messages.merge(
        messages_df[messages_df['tm'] == 1].drop(messages_df.columns.difference(['id', 'index']), axis=1),
        how='left', left_on='id', right_on='id')
    untagged_messages = untagged_messages.merge(
        messages_df[messages_df['tm'] == 0].drop(messages_df.columns.difference(['id', 'index']), axis=1), how='inner',
        left_on='id',
        right_on='id')

    messages_df.rename(columns={'id': '_id', 'index': 'id'}, inplace=True)
    tagged_messages.rename(columns={'id': '_id', 'index': 'id'}, inplace=True)
    untagged_messages.rename(columns={'id': '_id', 'index': 'id'}, inplace=True)

    edges, nodes = tt.create_graph(messages_df.copy())
    data_sim = pr.jaccard_sim(edges, nodes)

    graph = nx.from_pandas_edgelist(data_sim, source='source', target='target', edge_attr=True)
    # graph.remove_edges_from(nx.selfloop_edges(graph))
    node_dic = dict(zip(tagged_messages['id'], tagged_messages['tag']))
    nx.set_node_attributes(graph, node_dic, 'label')

    tagged_messages = tagged_messages[tagged_messages['id'].isin(graph.nodes)]
    untagged_messages = untagged_messages[untagged_messages['id'].isin(graph.nodes)]

    labels = ct.classification_graph(G=graph, weight='jaccard_sim', train_data=tagged_messages,
                                     test_data=untagged_messages, method=method, sub_method=sub_method,
                                     label_method=label_method, n_head_score=n_head_score)

    labels = labels.merge(untagged_messages, left_index=True, right_on='id')
    labels.reset_index(drop=True, inplace=True)
    labels.drop(['id', 'tm'], axis=1, inplace=True)
    labels.rename(columns={'_id': 'id'}, inplace=True)

    return labels
