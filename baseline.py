from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn import svm, linear_model
import argparse
import data_loader
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='lasso', type=str, help='lasso | rf | svm | random | random_small')



def classify(model, tf_train, tf_val, tf_test, train, valid, test):
    model.fit(tf_train, train['label'])
    val_pred = model.predict(tf_val)
    test_pred = model.predict(tf_test)
    print(accuracy_score(valid['label'], val_pred),
          accuracy_score(test['label'], test_pred))
    print(confusion_matrix(valid['label'], val_pred),
          confusion_matrix(test['label'], test_pred))


def main():
    train, valid, test = data_loader.load_data('train.txt', 'valid.txt', 'test.txt')
    print(len(train['text']), len(valid['text']), len(test['text']))
    print(train['text'][0])
    avg = np.mean([len(x.split(' ')) for x in train['text']])
    print(avg)
    assert False

    tf_vectorizer = CountVectorizer()
    tf_train = tf_vectorizer.fit_transform(train['text'])
    tf_val = tf_vectorizer.transform(valid['text'])
    tf_test = tf_vectorizer.transform(test['text'])
    models = [RandomForestClassifier(n_estimators = 100),
             svm.SVC(),
             linear_model.LogisticRegression(penalty='l2')]
    for model in models:
        classify(model, tf_train, tf_val, tf_test, train, valid, test)


if __name__ == "__main__":
    main()
