#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import keras
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from keras.datasets import imdb
from time import time

import re
import string
import nltk
from nltk.tokenize import word_tokenize
from string import punctuation 
from nltk.corpus import stopwords 
nltk.download('punkt')
nltk.download('stopwords')
import pandas as pd
from nltk.stem import PorterStemmer 
from nltk import FreqDist 
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from PyTsetlinMachineCUDA.tm import MultiClassTsetlinMachine
nltk.download('wordnet')
from time import time 
stop_words = set(stopwords.words('english'))
tokenizerR = RegexpTokenizer(r'\w+')
from numpy import save
from nltk.stem import WordNetLemmatizer 
from argparse import ArgumentParser
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap



parser = ArgumentParser()
parser.add_argument('-n_clauses_per_class', type=int, default=5000)
parser.add_argument('-s', type=float, default=5.0)
parser.add_argument('-T', type=int, default=80)
parser.add_argument('-drop_clause', type=float, default=0.0)
parser.add_argument('-state_bits', type=int, default=8)
parser.add_argument('-features', type=int, default=7500)
parser.add_argument('-gpus', type=int, default=16)
parser.add_argument('-stop_train', type=int, default=250)
parser.add_argument('-example', type=int, default=1)

config = parser.parse_args()

clauses = config.n_clauses_per_class
T = config.T
s = config.s
drop_clause = config.drop_clause
number_of_state_bits = config.state_bits
n_gpus = config.gpus

MAX_NGRAM = 3

NUM_WORDS=7500
INDEX_FROM=2 

FEATURES=config.features

print("Downloading dataset...")

train,test = keras.datasets.imdb.load_data(num_words=NUM_WORDS, index_from=INDEX_FROM)

train_x,train_y = train
test_x,test_y = test

word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

print("Producing bit representation...")

# Produce N-grams

id_to_word = {value:key for key,value in word_to_id.items()}

vocabulary = {}
for i in range(train_y.shape[0]):
	terms = []
	for word_id in train_x[i]:
		terms.append(id_to_word[word_id])
	
	for N in range(1,MAX_NGRAM+1):
		grams = [terms[j:j+N] for j in range(len(terms)-N+1)]
		for gram in grams:
			phrase = " ".join(gram)
			
			if phrase in vocabulary:
				vocabulary[phrase] += 1
			else:
				vocabulary[phrase] = 1

# Assign a bit position to each N-gram (minimum frequency 10) 

phrase_bit_nr = {}
bit_nr_phrase = {}
bit_nr = 0
for phrase in vocabulary.keys():
	if vocabulary[phrase] < 10:
		continue

	phrase_bit_nr[phrase] = bit_nr
	bit_nr_phrase[bit_nr] = phrase
	bit_nr += 1

# Create bit representation

X_train = np.zeros((train_y.shape[0], len(phrase_bit_nr)), dtype=np.uint32)
Y_train = np.zeros(train_y.shape[0], dtype=np.uint32)
for i in range(train_y.shape[0]):
	terms = []
	for word_id in train_x[i]:
		terms.append(id_to_word[word_id])

	for N in range(1,MAX_NGRAM+1):
		grams = [terms[j:j+N] for j in range(len(terms)-N+1)]
		for gram in grams:
			phrase = " ".join(gram)
			if phrase in phrase_bit_nr:
				X_train[i,phrase_bit_nr[phrase]] = 1

	Y_train[i] = train_y[i]

X_test = np.zeros((test_y.shape[0], len(phrase_bit_nr)), dtype=np.uint32)
Y_test = np.zeros(test_y.shape[0], dtype=np.uint32)

for i in range(test_y.shape[0]):
	terms = []
	for word_id in test_x[i]:
		terms.append(id_to_word[word_id])

	for N in range(1,MAX_NGRAM+1):
		grams = [terms[j:j+N] for j in range(len(terms)-N+1)]
		for gram in grams:
			phrase = " ".join(gram)
			if phrase in phrase_bit_nr:
				X_test[i,phrase_bit_nr[phrase]] = 1				

	Y_test[i] = test_y[i]

print("Selecting features...")

SKB = SelectKBest(chi2, k=FEATURES)
SKB.fit(X_train, Y_train)

selected_features = SKB.get_support(indices=True)
X_train = SKB.transform(X_train)
X_test = SKB.transform(X_test)

tm1 = MultiClassTsetlinMachine(clauses*2, T, s, clause_drop_p=drop_clause, number_of_state_bits=number_of_state_bits, number_of_gpus=n_gpus)

f = open("imdb_weighted_%.1f_%d_%d_%.2f_%d_aug.txt" % (s, clauses, T,  drop_clause, number_of_state_bits), "w+")

r_50 = 0
r_25 = 0
max = 0.0

for i in range(config.stop_train):
	start_training = time()
	tm1.fit(X_train, Y_train, epochs=1, incremental=True)
	stop_training = time()

	start_testing = time()
	result = 100*(tm1.predict(X_test) == Y_test).mean()
	stop_testing = time()
	if result > max:
		max = result
	
	if i >= 175:
     		r_25+=result
    
	if i >= 150:
     		r_50+=result

	print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs \n" % (i+1, result, stop_training-start_training, stop_testing-start_testing), file=f)

print("Average Accuracy last 50 epochs: %.2f \n" %(r_50/50), file=f)
print("Average Accuracy last 25 epochs: %.2f \n" %(r_25/25), file=f)
print("Max Accuracy: %.2f \n" %(max), file=f)

if config.interpret:
    example = config.example
    print('predicted Class:  ', tm1.predict(X_test[example:example+1,:]))
    triggClause = tm1.transform(X_test[example:example+1,:])
    clauseIndex = []
    for i in range(len(triggClause[0])):
        if triggClause[0][i] ==1:
            clauseIndex.append(i)

    import nltk
    from nltk.probability import FreqDist
    originalFeatures = []
    negatedFeatures = []

    number_of_features = 1000
    for j in range(0, 1500, 2):
            #print("Clause #%d (%d): " % (j, tm1.get_weight(1, j)), end=' ')
            l = []
            for k in range(number_of_features*2):
                    if tm1.ta_action(0, j, k) == 1:
                            if k < number_of_features:
                                    l.append(" x%d" % (k))
                                    originalFeatures.append(k)
                            else:
                                    l.append("¬x%d" % (k-number_of_features))
                                    negatedFeatures.append(k-number_of_features)
            #print(" ∧ ".join(l))
            
            
    fdist1 = FreqDist(negatedFeatures)
    negatedWords = fdist1.most_common(200)

    fdist2 = FreqDist(originalFeatures)
    originalWords = fdist2.most_common(20)

    print('full original word')
    fulloriginalword=[]
    for i in originalWords:
        fulloriginalword.append(i[0])
        
    fullnegatedword =[]
    print('full negated word')
    for i in negatedWords:
        fullnegatedword.append(i[0])


    originalFeatures2 = []
    negatedFeatures2= []

    for j in clauseIndex:
        if j < 1500 and j%2==0:
            #print("Clause #%d (%d): " % (j, tm1.get_weight(1, j)), end=' ')
            l = []
            for k in range(number_of_features*2):
                    if tm1.ta_action(0, j, k) == 1:
                            if k < number_of_features:
                                    l.append(" x%d" % (k))
                                    originalFeatures2.append(k)
                            else:
                                    l.append("¬x%d" % (k-number_of_features))
                                    negatedFeatures2.append(k-number_of_features)


    fdist3 = FreqDist(negatedFeatures2)
    negatedWords2 = fdist3.most_common(100)

    fdist4 = FreqDist(originalFeatures2)
    originalWords2 = fdist4.most_common(10)


    neededoriginalword =[]
    print('needed original word')
    for i in originalWords2:
        neededoriginalword.append(i[0])

    needednegatedword =[]
    print('needed negated word')
    for i in negatedWords2:
        needednegatedword.append(i[0])
        
        
    #Save fulloriginalword, fullnegatedword, neededoriginalword, or needednegatedword (Preferred needednegatedword for interpretability)
    interpretList = np.asarray(needednegatedword)
    np.savetxt('interpret_imdb.csv', interpretList, fmt='%s')

    df = pd.read_csv('interpret_imdb.csv', dtype=str, header=None)
    df1 = df.iloc[:,:]
    full1 = df.iloc[:,:].values
    #full1= np.reshape(full1,(10,20))

    index = np.arange(100)
    letter2num = {}
    for i in range(len(index)):
        letter2num[full1[i][0]] =i
    print(letter2num)
    df2 = pd.DataFrame(np.array( [letter2num[i] for i in df1.values.flat] ).reshape(df1.shape))

    print(df2)

    colors = ["white"]  # use hex colors here, if desired.
    cmap = ListedColormap(colors)


    full2 = df.iloc[:,:].values
    full2= np.reshape(full2,(10,10))

    full3 = df2.iloc[:,:].values
    full3= np.reshape(full3,(10,10))
    fig, ax = plt.subplots()
    ax.imshow(full3,cmap='YlOrBr_r')


    for i in range(len(full2)):
        for j in range(10):
            ax.text(j,i, full2[i,j], ha="center", va="center")
    plt.axis('off')
    ax.set_aspect(0.3)
    plt.grid(True)
    plt.show()
