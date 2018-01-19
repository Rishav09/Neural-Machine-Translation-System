#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 13:22:57 2018

@author: rishav
"""

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                import string
import re
from pickle import dump,load
from unicodedata import normalize
from numpy import array
from numpy.random import rand
from numpy.random import shuffle

def load_doc(filename):
	file = open(filename, mode='rt', encoding='utf-8')
	text = file.read()
	file.close()
	return text

def to_pairs(doc):
	lines = doc.strip().split('\n')
	pairs = [line.split('\t') for line in  lines]
	return pairs

def clean_pairs(lines):
	cleaned = list()
	re_print = re.compile('[^%s]' % re.escape(string.printable))
	table = str.maketrans('', '', string.punctuation)
	for pair in lines:
		clean_pair = list()
		for line in pair:
			line = normalize('NFD', line).encode('ascii', 'ignore')
			line = line.decode('UTF-8')
			line = line.split()
			line = [word.lower() for word in line]
			line = [word.translate(table) for word in line]
			line = [re_print.sub('', w) for w in line]
			line = [word for word in line if word.isalpha()]
			clean_pair.append(' '.join(line))
		cleaned.append(clean_pair)
	return array(cleaned)

def save_clean_data(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)


filename = 'deu.txt'
doc = load_doc(filename)
pairs = to_pairs(doc)
clean_pairs = clean_pairs(pairs)
save_clean_data(clean_pairs, 'english-german.pkl')
for i in range(100):
	print('[%s] => [%s]' % (clean_pairs[i,0], clean_pairs[i,1]))
    




