#!/usr/bin/python

import crash_on_ipy
import logging
import pickle
import random
import csv
import re
import pdb

from gensim import corpora, models
from NaiveBayesClassifier import NaiveBayesClassifier
from collections import defaultdict

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

    # transform string based documents into tokenID based vectors
    print 'Read Data'
    reader = csv.reader(open("../../dataset/small_samples/project_integrated_data.csv", 'rU'))
    # reader = csv.reader(open("../../dataset/small_samples/test_data.csv", 'rU'))
    pg2 = project_generator2(reader)

    print 'Divide data by resource_type'

    corpus = defaultdict(list)
    labels = defaultdict(list)

    for projectid, resource_type, doc, train_test, is_exciting in pg2:
        d = dictionary.doc2bow(doc)
        l = 1 if is_exciting == '1' else 0
        
        corpus[resource_type].append(d)
        labels[resource_type].append(l)

    # shuffle the data
    random.seed(123456)
    for k in corpus.keys():
        combined = zip(corpus[k], labels[k])
        random.shuffle(combined)
        corpus[k][:], labels[k][:] = zip(*combined)

    # serialize data
    # pickle.dump(corpus, open('corpus.dat', 'wb'))
    # pickle.dump(labels, open('labels.dat', 'wb'))

    # corpus = pickle.load(open('corpus.dat', 'r'))
    # labels = pickle.load(open('labels.dat', 'r'))

    print 'Train LDA Models'
    lda_models = {}
    for k in corpus.keys():
        lda_models[k] = models.ldamodel.LdaModel(corpus[k], id2word=dictionary, num_topics=100)
        lda_models[k].save('lda_models/' + k + '.model')

    # load LDA models
    for i in os.walk('lda_models'):
        print i[1] 

    print 'Split Train / Test Data'
    train_corpus = defaultdict(list)
    test_corpus = defaultdict(list)
    train_label = defaultdict(list)
    test_label = defaultdict(list)

    train_dat = defaultdict(list)
    test_dat = defaultdict(list)

    for k in corpus.keys():
        print k
        tmp_train = corpus[k][:int(len(corpus[k])*0.8)]
        tmp_test = corpus[k][int(len(corpus[k])*0.8):]

        train_corpus[k] = lda_models[k][tmp_train]
        test_corpus[k] = lda_models[k][tmp_test]
        train_label[k] = labels[k][:int(len(labels[k])*0.8)]
        test_label[k] = labels[k][int(len(labels[k])*0.8):]
        
        train_dat[k] = [(format_vector_as_dict(d), l) for (d, l) in zip(train_corpus[k].corpus, train_label[k])]
        test_dat[k] = [(format_vector_as_dict(d), l) for (d, l) in zip(test_corpus[k].corpus, test_label[k])]

    # training phase
    print 'Start training Naive Bayes Classifier'
    for k in train_dat.keys():
        classifier = NaiveBayesClassifier.train(train_dat[k])

        # test the accuracy
        print 'Testing'
        results = classifier.batch_classify([fs for (fs, l) in test_dat[k]])
        correct = [l==r for ((fs,l), r) in zip(test_dat[k], results)]
        if correct:
            acc = float(sum(correct))/len(correct)
        else:
            acc = 0
        
        print k, acc

    # translate feature
    # dataset = [(format_vector_as_dict(d), l) for (d, l) in zip(corpus, labels)]

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

    