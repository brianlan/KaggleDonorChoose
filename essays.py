#!/usr/bin/python

import logging
import csv
import re
from gensim import corpora

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def projectGenerator(csv_reader):
    for projectid, teacher_acctid, title, short_description, need_statement, essay in reader:
        if projectid != 'projectid':
            doc = re.sub("\xe2\x80\x99", "'", short_description + essay)
            doc = re.split(',|\.| |\n|\t|\r|\\\\n|"|!|;|:|\+|%|@|#|\d|`|~|\?|<|>|/|\(|\)', doc.lower())
            doc = [w for w in doc if w != '' and w != '-']

            # remove stop words
            doc = [w for w in doc if w not in stoplist]

            yield doc

def projectGenerator2(csv_reader):
    for projectid, teacher_acctid, title, short_description, need_statement, essay in reader:
        if projectid != 'projectid':
            doc = re.sub("\xe2\x80\x99", "'", short_description + essay)
            doc = re.split(',|\.| |\n|\t|\r|\\\\n|"|!|;|:|\+|%|@|#|\d|`|~|\?|<|>|/|\(|\)', doc.lower())
            doc = [w for w in doc if w != '' and w != '-']

            # remove stop words
            doc = [w for w in doc if w not in stoplist]

            yield projectid, doc


stoplist1 = set(stopword.replace('\n','') for stopword in open('stoplist1.txt'))
stoplist2 = set(stopword.replace('\n','') for stopword in open('stoplist2.txt'))

stoplist = stoplist1 | stoplist2

# read essays
reader = csv.reader(open("../dataset/small_samples/essays.csv", 'rU'))
pg = projectGenerator(reader)

# build dictionary
dictionary = corpora.Dictionary(essay for essay in pg)
once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]
dictionary.filter_tokens(once_ids) # remove stop words and words that appear only once
dictionary.compactify()
dictionary.save('essays.dict')
# print dictionary.token2id

# transform string based documents into tokenID based vectors
reader = csv.reader(open("../dataset/small_samples/essays.csv", 'rU'))
pg2 = projectGenerator2(reader)
projectid_list = []
essays = []
for projectid, doc in pg2:
    essays.append(dictionary.doc2bow(doc))
    projectid_list.append(projectid)

# print projectid_list

corpora.MmCorpus.serialize('essays.mm', essays) # store to disk, for later use

# dictionary = corpora.Dictionary.load('essays.dict')
# essays = corpora.MmCorpus('essays.mm')

