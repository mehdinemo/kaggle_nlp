import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from doctools import app as docapp
from doctools import prepare_data as pr
from doctools import text_tools as tt


def main():
    train = pd.read_csv(r'data/train.csv')
    test = pd.read_csv(r'data/test.csv')

    train.drop(train.columns.difference(['id', 'text', 'target']), axis=1, inplace=True)
    test.drop(test.columns.difference(['id', 'text']), axis=1, inplace=True)

    train.rename(columns={'target': 'tag'}, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(train[['text', 'id', 'tag']], train['tag'], random_state=0,
                                                        test_size=0.3)

    edges, nodes = tt.create_graph(X_train.copy())
    data_sim = pr.jaccard_sim(edges, nodes)
    data_sim, sim_df = pr.sim_nodes_detector(data_sim)
    sim_df.reset_index(inplace=True)

    sim_df = pd.merge(sim_df, X_train[['id', 'tag']], how='left', left_on='index', right_on='id')
    sim_df_gr = sim_df.groupby('id_x').agg({'tag': ['sum', 'count']})
    sim_df_gr.columns = sim_df_gr.columns.droplevel()

    noise_data = list(sim_df_gr[(sim_df_gr['sum'] != 0) & (sim_df_gr['sum'] != sim_df_gr['count'])].index)
    noise_data.extend(sim_df[sim_df['id_x'].isin(noise_data)]['index'].values)

    X_train = X_train[~X_train['id'].isin(noise_data)]

    method = ''
    sub_method = ''
    label_method = 'louvain'
    n_head_score = 0.7
    messages = docapp.document_classification(tagged_messages=X_train, untagged_messages=X_test,
                                              method=method,
                                              sub_method=sub_method,
                                              label_method=label_method,
                                              n_head_score=n_head_score)

    tmp = X_test[['id', 'tag']].merge(messages[['id', 'label']], how='left', on='id')
    tmp.fillna(0, inplace=True)
    acc = classification_report(y_true=tmp['tag'], y_pred=tmp['label'])

    # messages.dropna(inplace=True)
    # messages['label'] = messages['label'].astype(int)
    # acc = classification_report(messages['tag'], messages['label'])
    print(acc)

    # messages.to_excel('messages_louvain.xlsx', index=False)
    # print('done')


if __name__ == '__main__':
    main()
