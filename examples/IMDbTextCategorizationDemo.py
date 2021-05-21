#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import keras
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from keras.datasets import imdb
from pyTsetlinMachine.tm import MultiClassTsetlinMachine
from time import time

MAX_NGRAM = 3

NUM_WORDS=7500
INDEX_FROM=2 

FEATURES=7500

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

tm = MultiClassTsetlinMachine(15000, 80*100, 10.0, weighted_clauses=True, clause_drop_p=0.75, literal_drop_p=0.25)
r_50 = 0
r_25 = 0
max = 0.0

print("\nAccuracy over 400 epochs:\n")
for i in range(400):
	start_training = time()
	tm.fit(X_train, Y_train, epochs=1, incremental=True)
	stop_training = time()

	start_testing = time()
	result = 100*(tm.predict(X_test) == Y_test).mean()
	stop_testing = time()
	if result > max:
		max = result
	
	if i >= 175:
     		r_25+=result
    
	if i >= 150:
     		r_50+=result

	print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs \n" % (i+1, result, stop_training-start_training, stop_testing-start_testing))
	file = open('logger_imdbTM_dc_0.75_dl_0.25_c_10k_s_10_f7500.txt', 'a')
	file.writelines("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs \n" % (i+1, result, stop_training-start_training, stop_testing-start_testing))
	file.close()

print("Average Accuracy last 50 epochs: %.2f \n" %(r_50/50))
print("Average Accuracy last 25 epochs: %.2f \n" %(r_25/25))
print("Max Accuracy: %.2f \n" %(max))

file = open('logger_imdbTM_dc_0.75_dl_0.25_c_10k_s_10_f7500.txt', 'a')
file.writelines("Average Accuracy last 50 epochs: %.2f \n" %(r_50/50))
file.writelines("Average Accuracy last 25 epochs: %.2f \n" %(r_25/25))
file.writelines("Max Accuracy: %.2f \n" %(max))
file.close()

state = tm.get_state()

#save/load the state to file using numpy save/load.

np.savez_compressed("imdbTM_dc_0.75_dl_0.25_c_10k_s_10_f7500.npz", state)
#state = np.load("filename.npz")['arr_0']

#set the state back by calling

#tm.set_state(state).

#and then

#tm.fit(X_train, Y_train, epochs=0)


