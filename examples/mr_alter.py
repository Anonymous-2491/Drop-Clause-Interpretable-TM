import re
import string
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from string import punctuation 
from nltk.corpus import stopwords 
nltk.download('punkt')
nltk.download('stopwords')
import pandas as pd
from nltk.stem import PorterStemmer 
from nltk import FreqDist 
from nltk.tokenize import RegexpTokenizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from PyTsetlinMachineCUDA.tm import MultiClassTsetlinMachine
nltk.download('wordnet')
from time import time 
stop_words = set(stopwords.words('english'))
tokenizerR = RegexpTokenizer(r'\w+')
from numpy import save
from nltk.stem import WordNetLemmatizer 
stop_words = set(stopwords.words('english'))
alpha = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

from argparse import ArgumentParser
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import gensim.downloader

'''
'fasttext-wiki-news-subwords-300',
 'conceptnet-numberbatch-17-06-300',
 'word2vec-ruscorpora-300',
 'word2vec-google-news-300',
 'glove-wiki-gigaword-50',
 'glove-wiki-gigaword-100',
 'glove-wiki-gigaword-200',
 'glove-wiki-gigaword-300',
 'glove-twitter-25',
 'glove-twitter-50',
 'glove-twitter-100',
 'glove-twitter-200',
'''
#print(gensim.downloader.info())
glove_vectors = gensim.downloader.load('glove-wiki-gigaword-300')
#print(glove_vectors.wv)
print("glove-wiki-gigaword-300")

parser = ArgumentParser()
parser.add_argument('-interpret', type=bool, default=False)
parser.add_argument('-n_clauses_per_class', type=int, default=10000)
parser.add_argument('-s', type=float, default=2.0)
parser.add_argument('-T', type=int, default=80)
parser.add_argument('-drop_clause', type=float, default=0.75)
parser.add_argument('-drop_literal', type=float, default=0.5)
parser.add_argument('-state_bits', type=int, default=8)
parser.add_argument('-features', type=int, default=10000)
parser.add_argument('-gpus', type=int, default=1)
parser.add_argument('-stop_train', type=int, default=250)
parser.add_argument('-example', type=int, default=1)

config = parser.parse_args()

clauses = config.n_clauses_per_class
c = clauses*2
T = config.T
s = config.s
drop_clause = config.drop_clause
drop_literal = config.drop_literal
number_of_state_bits = config.state_bits
n_gpus = config.gpus

config = parser.parse_args()
alter = True
tt = 7108

df = pd.read_csv('MR.csv', delimiter=';')
label = df.iloc[:,1:2].values
textOrig = df.iloc[:,0:1].values
y = np.reshape(label, len(label))
#print(textOrig.shape)
#print(textOrig[0])

print(glove_vectors.most_similar('awful', topn=10))

p = open("mr_altered_text_glovewiki.txt", "w+")

def prepreocess(data):

    input_data=[]
    vocab   = []
    for i in data:
        for j in i:
            j = j.lower()
            j = j.replace("\n", "")
            j = j.replace('n\'t', 'not')
            j = j.replace('\'ve', 'have')
            j = j.replace('\'ll', 'will')
            j = j.replace('\'re', 'are')
            j = j.replace('\'m', 'am')
            j = j.replace('/', ' / ')
            j = j.replace('-', ' ')
            j = j.replace('!', ' ')
            j = j.replace('?', ' ')
            j = j.replace('+', ' ')
            j = j.replace('*', ' ')
            while "  " in j:
                j = j.replace('  ', ' ')
            while ",," in j:
                j = j.replace(',,', ',')
            j = j.strip()
            j = j.strip('.')
            j = j.strip()

            temp1 = tokenizerR.tokenize(j)
            temp2 = [x for x in temp1 if not x.isdigit()]
            temp3 = [w for w in temp2 if not w in alpha]
            temp4 = [w for w in temp3 if not w in stop_words]
            input_data.append(temp4)
    return input_data

input_text = prepreocess(textOrig)
'''
inputtext = []
for i in input_text:
    ps = PorterStemmer()
    temp4 = []
    for m in i:
        temp_temp =ps.stem(m)
        temp4.append(temp_temp)
    inputtext.append(temp4)
'''
newVocab =[]
for i in input_text:
    for j in i:
            newVocab.append(j)

fdist1 = FreqDist(newVocab)
tokens1 = fdist1.most_common(config.features)

full_token_fil = []
for i in tokens1:
    full_token_fil.append(i[0])

sum1 = 0
for j in tokens1:
    sum1 += j[1]

vocab_unique = full_token_fil
vocab = np.asarray(full_token_fil)
np.savetxt('mr_vocab.csv', vocab, delimiter=',', fmt='%s')

def binarization_text(data4):
    feature_set = np.zeros([len(data4), config.features], dtype=np.uint8)
    tnum=0
    for t in data4:
        for w in t:
            if (w in vocab_unique):
                idx = vocab_unique.index(w)
                feature_set[tnum][idx] = 1
        tnum += 1
    return feature_set

def alter_data(data):
    for i in range(tt,len(data)):
        sim_valid = []
        print("#", i, file=p)
        print("Sentence:", data[i], file=p)
        if random.random() > 0.5:
            print("Alteration!", file=p)
            word = random.choice(data[i])
            print("Word:", word, file=p)
            try:
                sims = glove_vectors.most_similar(word, topn=10)  # get other similar words
            except KeyError:
                continue
            print("Similar words:", sims, file=p)
            for j in range(len(sims)):
                if sims[j][0] in vocab:
                    sim_valid.append(sims[j][0])
            print("Alternate word:", sim_valid, file=p)
            if len(sim_valid):
                for k in range(len(data[i])):
                    if data[i][k] == word:
                        data[i][k] = sim_valid[0]
                        break
        print("Altered Sentence:", data[i], file=p)
    return data

print("_________________________________________________________________________________")
if alter:
    inputtext = alter_data(input_text)
else:
    inputtext = input_text
X_text = binarization_text(inputtext)
print("Text length:", X_text.shape)

X_train = X_text[0:tt,:]
print("X_train length:", X_train.shape)
X_test  = X_text[tt:,:]
print("X_test length:", X_test.shape)
ytrain = y[0:tt]
ytest = y[tt:]
X_dev = X_text[tt:,:]
Y_dev = y[tt:]

tm1 = MultiClassTsetlinMachine(c, T*100, s, clause_drop_p=drop_clause, feature_drop_p=drop_literal, number_of_gpus=n_gpus, number_of_state_bits=number_of_state_bits)

f = open("mr_weighted_%.1f_%d_%d_%.2f_%.2f_%d_10kf_robust.txt" % (s, clauses, T, drop_clause, drop_literal, number_of_state_bits), "w+")

r_25 = 0.0
max = 0.0

for i in range(config.stop_train):
	start_training = time()
	tm1.fit(X_train, ytrain, epochs=1, incremental=True)
	stop_training = time()

	start_testing = time()
	result2 = 100*(tm1.predict(X_train) == ytrain).mean()
	result1 = 100*(tm1.predict(X_test) == ytest).mean()
	#result1 = 0
	stop_testing = time()

	if result1 > max:
		max = result1

	if i >= config.stop_train-25:
		r_25+=result1

	print("#%d AccuracyTrain: %.2f%% AccuracyTest: %.2f%%  Training: %.2fs Testing: %.2fs" % (i+1, result2, result1, stop_training-start_training, stop_testing-start_testing), file=f)
	print("#%d AccuracyTrain: %.2f%% AccuracyTest: %.2f%%  Training: %.2fs Testing: %.2fs" % (i+1, result2, result1, stop_training-start_training, stop_testing-start_testing))
	f.flush()

