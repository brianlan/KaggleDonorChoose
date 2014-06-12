#!/usr/bin/python

from gensim import corpora

dictionary = corpora.Dictionary.load('essays.dict')
print len(dictionary.token2id)