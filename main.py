import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from doctools import app as docapp


def main():
    train = pd.read_csv(r'data/train.csv')
    test = pd.read_csv(r'data/test.csv')

    train.drop(train.columns.difference(['id', 'text', 'target']), axis=1, inplace=True)
    test.drop(test.columns.difference(['id', 'text']), axis=1, inplace=True)

    train.rename(columns={'target': 'tag'}, inplace=True)

    # X_train, X_test, y_train, y_test = train_test_split(train[['text', 'id', 'tag']], train['tag'], random_state=0,
    #                                                     test_size=0.3)

    method = 'degree'
    sub_method = 'degree'
    label_method = 'max'
    n_head_score = 0.5
    messages = docapp.document_classification(tagged_messages=train, untagged_messages=test,
                                              method=method,
                                              sub_method=sub_method,
                                              label_method=label_method,
                                              n_head_score=n_head_score)

    messages.dropna(inplace=True)
    acc = classification_report(messages['tag'], messages['label'])
    print(acc)


if __name__ == '__main__':
    main()
