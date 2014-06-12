#!/usr/bin/python

import crash_on_ipy
import logging
import pickle
import random
import csv
import re

from gensim import corpora
from gensim import models
from NaiveBayesClassifier import NaiveBayesClassifier

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def project_generator(csv_reader):
    for projectid, resource_type, essay, train_test, is_exciting in csv_reader:
        if projectid != 'projectid':
            doc = re.sub("\xe2\x80\x99", "'", essay)
            doc = re.split(',|\.| |\n|\t|\r|\\\\n|"|!|;|:|\+|%|@|#|\d|`|~|\?|<|>|/|\(|\)', doc.lower())
            doc = [w for w in doc if w != '' and w != '-']

            # remove stop words
            doc = [w for w in doc if w not in stoplist]

            yield doc


def project_generator2(csv_reader):
    for projectid, resource_type, essay, train_test, is_exciting in csv_reader:
        if projectid != 'projectid':
            doc = re.sub("\xe2\x80\x99", "'", essay)
            doc = re.split(',|\.| |\n|\t|\r|\\\\n|"|!|;|:|\+|%|@|#|\d|`|~|\?|<|>|/|\(|\)', doc.lower())
            doc = [w for w in doc if w != '' and w != '-']

            # remove stop words
            doc = [w for w in doc if w not in stoplist]

            yield projectid, resource_type, doc, train_test, is_exciting


def format_vector_as_dict(vec):
    d = {}
    for w in vec:
        d[w[0]] = w[1]
    return d


if __name__ == '__main__':
    stoplist1 = set(stopword.replace('\n','') for stopword in open('stoplist1.txt'))
    stoplist2 = set(stopword.replace('\n','') for stopword in open('stoplist2.txt'))

    stoplist = stoplist1 | stoplist2

    # # read essays
    # reader = csv.reader(open("../../dataset/small_samples/project_integrated_data.csv", 'rU'))
    # pg = project_generator(reader)

    # # build dictionary
    # dictionary = corpora.Dictionary(essay for essay in pg)
    # once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]
    # dictionary.filter_tokens(once_ids) # remove stop words and words that appear only once
    # dictionary.compactify()
    # dictionary.save('essays.dict')

    # print dictionary.token2id
    dictionary = corpora.Dictionary.load('essays.dict')

    # # transform string based documents into tokenID based vectors
    reader = csv.reader(open("../../dataset/small_samples/project_integrated_data.csv", 'rU'))
    pg2 = project_generator2(reader)

    corpus = []
    labels = []

    test_set = []
    for projectid, resource_type, doc, train_test, is_exciting in pg2:
        if train_test == 'train':
            d = dictionary.doc2bow(doc)
            corpus.append(d)
            labels.append(1 if is_exciting == '1' else 0)
        else:
            test_set.append(format_vector_as_dict)


    # original feature
    dataset = [(format_vector_as_dict(d), l) for (d, l) in zip(corpus, labels)]

    # TFIDF
    # tfidf = models.TfidfModel(corpus)
    # dataset = []
    # for (d, l) in zip(corpus, labels):
    #     dataset.append((format_vector_as_dict(tfidf[d]), l))

    # serialize train / test set
    # pickle.dump(dataset, open('dataset.dat', 'wb'))
    # pickle.dump(test_set, open('test_set.dat', 'wb'))

    # dataset = pickle.load(open('dataset.dat', 'r'))
    # test_set = pickle.load(open('test_set.dat', 'r'))

    # split original training set into train / test set.
    print 'Prepare Train / Test Data'
    random.seed(123456)
    random.shuffle(dataset)
    train_dat = dataset[:int(len(dataset)*0.8)]
    test_dat = dataset[int(len(dataset)*0.8):]

    

    len_train = len(train_dat)
    len_test = len(test_dat)
    num_train0 = sum([l == 0 for t, l in train_dat])
    num_train1 = sum([l == 1 for t, l in train_dat])
    num_test0 = sum([l == 0 for t, l in test_dat])
    num_test1 = sum([l == 1 for t, l in test_dat])

    print len_train, len_test, num_train0, num_train1, num_test0, num_test1

    # training phase
    print 'Start training Naive Bayes Classifier'
    classifier = NaiveBayesClassifier.train(train_dat)

    # test the accuracy
    print 'Testing'
    results = classifier.batch_classify([fs for (fs, l) in test_dat])
    correct = [l==r for ((fs,l), r) in zip(test_dat, results)]
    if correct:
        acc = float(sum(correct))/len(correct)
    else:
        acc = 0

    print acc