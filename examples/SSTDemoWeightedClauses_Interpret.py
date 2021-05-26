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



parser = ArgumentParser()
parser.add_argument('-interpret', type=bool, default=False)
parser.add_argument('-n_clauses_per_class', type=int, default=5000)
parser.add_argument('-s', type=float, default=5.0)
parser.add_argument('-T', type=int, default=80)
parser.add_argument('-drop_clause', type=float, default=0.0)
parser.add_argument('-state_bits', type=int, default=8)
parser.add_argument('-features', type=int, default=7500)
parser.add_argument('-gpus', type=int, default=1)
parser.add_argument('-stop_train', type=int, default=250)
parser.add_argument('-example', type=int, default=1)

clauses = config.n_clauses_per_class
T = config.T
s = config.s
drop_clause = config.drop_clause
number_of_state_bits = config.state_bits
n_gpus = config.gpus

config = parser.parse_args()

col_list = ["text", "label"]
df = pd.read_csv('sst2.csv')
label = df.iloc[:,0:1].values
textOrig = df.iloc[:,1:2].values
y = np.reshape(label, len(label))
print(textOrig.shape)

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
            #temp4 = [w for w in temp3 if not w in stop_words]

            input_data.append(temp3)
 
    return input_data

input_text = prepreocess(textOrig)


inputtext = []
for i in input_text:
    ps = PorterStemmer()
    temp4 = []
    for m in i:
        temp_temp =ps.stem(m)
        temp4.append(temp_temp)
    inputtext.append(temp4)

newVocab =[]
for i in inputtext:
    for j in i:
            newVocab.append(j)

print(len(newVocab))
fdist1 = FreqDist(newVocab)
tokens1 = fdist1.most_common(config.features)


full_token_fil = []
for i in tokens1:
    full_token_fil.append(i[0])

sum1 = 0
for j in tokens1:
    sum1 += j[1]

print('sum1', sum1)

vocab_unique = full_token_fil
vocab = np.asarray(full_token_fil)
np.savetxt('sst_vocab.csv', vocab, delimiter=',', fmt='%s')

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

X_text = binarization_text(inputtext)

print("Text length:", X_text.shape)

tt = 6920

X_train = X_text[0:tt,:]
print("X_train length:", X_train.shape)
X_test  = X_text[tt:,:]
print("X_test length:", X_test.shape)
ytrain = y[0:tt]
ytest = y[tt:]
print(ytest.shape)
X_dev = X_text[tt:,:]
Y_dev = y[tt:]

tm1 = MultiClassTsetlinMachine(clauses*2, T*16, s, clause_drop_p=drop_clause, number_of_gpus=n_gpus, number_of_state_bits=number_of_state_bits)

f = open("sst_weighted_%.1f_%d_%d_%.2f_%d_aug.txt" % (s, clauses, T, drop_clause, number_of_state_bits), "w+")

r_25 = 0
r_50 = 0

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

	if i >= 350:
		r_50+=result1

	if i >= 375:
		r_25+=result1

	print("#%d AccuracyTrain: %.2f%% AccuracyTest: %.2f%%  Training: %.2fs Testing: %.2fs" % (i+1, result2, result1, stop_training-start_training, stop_testing-start_testing), file=f)

print("Average Accuracy last 25 epochs: %.2f \n" %(r_25/25), file=f)
print("Average Accuracy last 50 epochs: %.2f \n" %(r_50/50), file=f)
print("Max Accuracy: %.2f \n" %(max), file=f)

if config.interpret:
    example = config.example
    print('predicted Class:  ', tm1.predict(X_train[example:example+1,:]))
    triggClause = tm1.transform(X_train[example:example+1,:])
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
    np.savetxt('interpretFile.csv', interpretList, fmt='%s')

    df = pd.read_csv('interpretFile.csv', dtype=str, header=None)
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
