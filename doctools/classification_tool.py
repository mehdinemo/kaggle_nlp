import math
import networkx as nx
import pandas as pd


def classification_graph(G: nx.Graph, train_data: pd.DataFrame, test_data: pd.DataFrame, weight='weight', method='',
                         sub_method='', label_method='mean', n_head_score=1):
    # labels = nx.get_node_attributes(G, 'label')
    # labels = pd.DataFrame.from_dict(labels, orient='index')
    # labels.reset_index(inplace=True)
    # labels.columns = ['node', 'class']

    if method == '':
        scores_train = pd.DataFrame()
    else:
        G_train = G.subgraph(train_data['id'])
        scores_train = scores_degree(G_train, weight, method=method, sub_method=sub_method)

        scores_train.sort_values(by=['class', 'score'], ascending=False, inplace=True)
        classes = scores_train['class'].unique()
        scores_sorted = pd.DataFrame()
        for c in classes:
            c_score = scores_train[scores_train['class'] == c].copy()
            c_score.sort_values(by=['class', 'score'], ascending=False, inplace=True)
            n = math.ceil(n_head_score * len(c_score))
            c_score = c_score.head(n)
            scores_sorted = scores_sorted.append(c_score)

        scores_train = scores_sorted

    # labels.set_index('node', inplace=True)
    # adjacency matrix
    sim = adj_matrix(G, weight)

    sim_test_train = sim.drop(list(train_data['id']))
    sim_test_train.drop(columns=list(test_data['id']), axis=1, inplace=True)
    test_predict = fit_nodes(G, weight, sim_test_train, train_data, scores_train, label_method)

    return test_predict
    # pr.print_results(test_predict, labels)


def scores_degree(G: nx.Graph, weight='weight', method='degree', sub_method='degree') -> pd.DataFrame:
    if method == 'eig':
        degrees_df = nx.eigenvector_centrality(G, weight=weight)
        degrees_df = pd.DataFrame.from_dict(degrees_df, orient='index')
        degrees_df.reset_index(inplace=True)
        degrees_df.columns = ['node', 'degree']
    elif method == 'degree':
        degrees_df = nx.degree(G, weight=weight)
        degrees_df = pd.DataFrame(degrees_df)
        degrees_df.columns = ['node', 'degree']

    classes = pd.DataFrame(nx.get_node_attributes(G, 'label').items(), columns=['node', 'class'])
    degrees_df = degrees_df.merge(classes, how='left', left_on='node', right_on='node')

    classes = classes['class'].unique()

    # create subgragps for each class
    subgraph_dic = {}
    for i in classes:
        sub_nodes = (
            node
            for node, data
            in G.nodes(data=True)
            if data.get('label') == i
        )
        subgraph = G.subgraph(sub_nodes)
        subgraph_dic.update({i: subgraph})

    # calculate degree for nodes in subgraphs
    sub_deg_df = pd.DataFrame()
    for k, v in subgraph_dic.items():
        if sub_method == 'eig':
            sub_deg = nx.eigenvector_centrality(v, max_iter=200, weight=weight)
            sub_deg = pd.DataFrame.from_dict(sub_deg, orient='index')
            sub_deg.reset_index(inplace=True)
            sub_deg.columns = ['node', 'class_degree']
            sub_deg['class_degree'] = sub_deg['class_degree'] / sub_deg['class_degree'].sum()
        elif sub_method == 'degree':
            sub_deg = nx.degree(v, weight=weight)
            sub_deg = pd.DataFrame(sub_deg)
            sub_deg.columns = ['node', 'class_degree']

        sub_deg_df = sub_deg_df.append(sub_deg)

    degrees_df = degrees_df.merge(sub_deg_df, how='left', left_on='node', right_on='node')

    if method == 'eig':
        degrees_df['class_degree'] = degrees_df['class_degree'] / degrees_df['class_degree'].sum()
        degrees_df['degree'] = degrees_df['degree'] / degrees_df['degree'].sum()

    degrees_df['score'] = degrees_df['class_degree'] / degrees_df['degree']

    degrees_df.drop(degrees_df.columns.difference(['node', 'class', 'score']), axis=1, inplace=True)

    return degrees_df


def fit_nodes(G, weight, sim, labels: pd.DataFrame, scores=pd.DataFrame(), nscore_method='mean') -> pd.DataFrame:
    labels.rename(columns={'tag': 'class'}, inplace=True)

    G_train = G.subgraph(sim.columns)

    degree = G_train.degree(weight=weight)
    degree = pd.DataFrame(degree)
    degree = degree.merge(labels['class'], how='left', left_on=0, right_index=True)
    k_c = degree.groupby(['class'])[1].sum()
    k_c = k_c / 2
    m = k_c.sum()

    if not scores.empty:
        scores.set_index('node', inplace=True)
    sim = pd.DataFrame(sim)

    predict = pd.DataFrame(columns=['node', 'label'])
    for index, row in sim.iterrows():
        row = row.to_frame()
        if scores.empty:
            row = row.merge(labels[['id', 'class']], how='left', left_index=True, right_on='id')
        else:
            row = row.merge(scores, how='left', left_index=True, right_index=True)
            # row[index] = row[index] * row['score']

        row.dropna(inplace=True)
        # row = row[row[index] != 0]

        if nscore_method == 'louvain':
            if len(row) > 0:
                a_ij = row.groupby(['class'])[index].sum()
                k_i = row[index].sum()
                k_j = k_c + a_ij

                modularity = (a_ij - ((k_i * k_j) / (2 * (m + k_i)))) / (2 * (m + k_i))

                n_label = modularity.idxmax()
            else:
                n_label = None
        elif nscore_method == 'max':
            if len(row) > 0:
                try:
                    ind_max = row[index].idxmax()
                    n_score = row.loc[ind_max]
                    n_label = n_score['class']
                except Exception as ex:
                    print(ex)
            else:
                n_label = None
        else:
            if nscore_method == 'sum':
                n_score = row.groupby(['class'])[index].sum()
            elif nscore_method == 'mean':
                n_score = row.groupby(['class'])[index].sum()
            duplicated_labels = n_score.duplicated(False)
            if (True in duplicated_labels.values) or (len(n_score) == 0):
                n_label = None
            else:
                n_label = n_score.idxmax()
        predict = predict.append({'node': index, 'label': n_label}, ignore_index=True)

    predict.set_index('node', inplace=True)

    return predict


def adj_matrix(G: nx.Graph, weight='weight'):
    all_nodes = list(G.nodes)
    sim = nx.to_numpy_array(G, weight=weight)

    sim = pd.DataFrame(sim)
    sim.index = all_nodes
    sim.columns = all_nodes

    return sim
