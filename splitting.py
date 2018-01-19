#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:54:32 2018

@author: rishav
"""
from pickle import load
from pickle import dump
from numpy.random import rand
from numpy.random import shuffle
def load_clean_sentences(filename):
    return load(open(filename,'rb'))

def save_clean_data(sentences,filename):
    dump(sentences,open(filename,'wb'))
    print('Saved: %s' %filename)

raw_dataset=load_clean_sentences('english-german.pkl')

n_sentences=10000
dataset=raw_dataset[:n_sentences,: ]
shuffle(dataset)
train,test=dataset[:9000],dataset[9000:]

save_clean_data(dataset, 'english-german-both.pkl')
save_clean_data(train, 'english-german-train.pkl')
save_clean_data(test, 'english-german-test.pkl')

