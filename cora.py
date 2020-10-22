import networkx as nx
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import time
from doctools import prepare_data as pr
from doctools import classification_tool as ct


# import numpy as np
# import matplotlib.pyplot as plt
# import numpy.linalg


def prepare_graph():
    cites = pd.read_csv(r'data\cites.csv', sep='\t', header=None)
    cites.columns = ['cited', 'citing']

    content = pd.read_csv(r'data\content.csv', sep='\t', header=None)
    # content.set_index(0, inplace=True)
    content.rename(columns={0: 'paper_id', content.columns[-1]: 'classes'}, inplace=True)
    classes = content[['paper_id', 'classes']].copy()
    content.drop(['classes'], axis=1, inplace=True)

    allkeywords = content.melt('paper_id', var_name='word', value_name='count')
    allkeywords = allkeywords[allkeywords['count'] != 0]
    allkeywords.drop(['count'], axis=1, inplace=True)

    graph = allkeywords.merge(allkeywords, how='inner', left_on='word', right_on='word')

    start_time = time.time()
    graph = graph.groupby(['paper_id_x', 'paper_id_y'], as_index=False)['word'].sum()
    end_time = time.time()

    print(end_time - start_time)

    graph.columns = ['source', 'target', 'weight']

    graph.to_csv('cora_graph.csv', index=False)


def prepare_data():
    cites = pd.read_csv(r'data\cites.csv', sep='\t', header=None)
    cites.columns = ['source', 'target']
    # G_cite = nx.from_pandas_edgelist(cites, source='source', target='target')
    G_cite = nx.from_pandas_edgelist(cites, source='source', target='target', create_using=nx.DiGraph())

    # adjacency matrix
    sim_cite = pr.adj_matrix(G_cite)

    content = pd.read_csv(r'data\content.csv', sep='\t', header=None)
    content.rename(columns={0: 'id', content.columns[-1]: 'class'}, inplace=True)
    classes = content[['id', 'class']].copy()

    graph = pd.read_csv(r'data\cora_graph.csv')
    nodes = graph[graph['source'] == graph['target']].copy()
    nodes.drop(['target'], axis=1, inplace=True)
    nodes.columns = ['node', 'node_weight']
    nodes.reset_index(drop=True, inplace=True)
    # nodes = content.drop(['id', 'class'], axis=1).sum()

    graph = graph[graph['source'] < graph['target']]
    graph.reset_index(drop=True, inplace=True)

    data_sim = pr.jaccard_sim(graph, nodes)

    # print('delete similar data...')
    # data_sim, sim_df = pr.sim_nodes_detector(data_sim)

    print('creating graph...')
    G = nx.from_pandas_edgelist(data_sim, source='source', target='target', edge_attr=True)
    # G.remove_edges_from(nx.selfloop_edges(G))
    print(f'graph created with {len(G)} nodes and {G.number_of_edges()} edges.')

    # node_dic = dict(zip(classes['id'], classes['class']))
    # nx.set_node_attributes(G, node_dic, 'label')

    # L = nx.normalized_laplacian_matrix(G, weight='jaccard_sim')
    # e = numpy.linalg.eigvals(L.A)
    # print("Largest eigenvalue:", max(e))
    # print("Smallest eigenvalue:", min(e))
    # plt.hist(e, bins=100)  # histogram with 100 bins
    # plt.xlim(0, 2)  # eigenvalues between 0 and 2
    # plt.show()

    # adjacency matrix
    sim = pr.adj_matrix(G, 'jaccard_sim')

    # merge two sim
    # sim_cite[sim_cite == 0] = 0.5
    sim_multiply = sim + sim_cite
    sim_multiply.drop(sim_cite.columns.difference(sim.columns), axis=1, inplace=True)
    sim_multiply.dropna(axis=0, inplace=True)
    G = nx.from_pandas_adjacency(sim_multiply)

    node_dic = dict(zip(classes['id'], classes['class']))
    nx.set_node_attributes(G, node_dic, 'label')

    labels = nx.get_node_attributes(G, 'label')
    labels = pd.DataFrame.from_dict(labels, orient='index')
    labels.reset_index(inplace=True)
    labels.columns = ['id', 'tag']

    X_train, X_test, y_train, y_test = train_test_split(labels, labels['tag'], test_size=0.3, random_state=0)
    # G_train = G.subgraph(X_train)

    # n = math.ceil(0.15 * len(G))
    # test_predict = pr.fit_nodes(sim, labels, scores, label_method)

    method = ''
    sub_method = ''
    label_method = 'louvain'
    n_head_score = 0.4
    test_predict = ct.classification_graph(G=G, weight='jaccard_sim', train_data=X_train,
                                           test_data=X_test, method=method, sub_method=sub_method,
                                           label_method=label_method, n_head_score=n_head_score)

    # test_predict.fillna(-1, inplace=True)
    test_predict.dropna(inplace=True)
    test_predict = pd.merge(test_predict, labels, how='left', left_index=True, right_on='id')
    # test_predict.to_csv('data/all_sample_predict.csv')
    acc = classification_report(y_true=test_predict['tag'], y_pred=test_predict['label'], output_dict=False)
    print(acc)


def main():
    # pr = PrepareData()
    # data = pd.read_csv(r'C:\Users\m.nemati\Desktop\nodes.csv')
    # g_data = data.groupby(['modularity_class', 'target'], as_index=False)['Id'].count()

    # features, adj, samples, labels = pr._load_data()

    # G = nx.from_scipy_sparse_matrix(adj)

    prepare_data()
    print('done')


if __name__ == '__main__':
    # import sklearn
    # print(sklearn.__version__)
    main()
