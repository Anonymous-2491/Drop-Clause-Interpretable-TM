from PyTsetlinMachineCUDA.tm import TsetlinMachine, MultiClassTsetlinMachine
import numpy as np 
from sympy.logic.boolalg import to_dnf, to_cnf
from sympy.logic.inference import satisfiable
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
import os
import cv2
'''
from pysat.formula import CNF
from pysat.solvers import Minisat22
from pysat.card import CardEnc, EncType

number_of_features = 10
noise = 0.1

X_train = np.random.randint(0, 2, size=(5000, number_of_features), dtype=np.uint32)
Y_train = np.logical_xor(X_train[:,0], X_train[:,1]).astype(dtype=np.uint32)
Y_train = np.where(np.random.rand(5000) <= noise, 1-Y_train, Y_train) # Adds noise

X_test = np.random.randint(0, 2, size=(5000, number_of_features), dtype=np.uint32)
Y_test = np.logical_xor(X_test[:,0], X_test[:,1]).astype(dtype=np.uint32)
'''
data_dir = r'./mnist_c'  #Change the data_dir to where you extracted the mnist_c.zip
#Simply comment out the corruptions that you don't want to use. Remember to keep the 'identity' as it is the actual test_images.
CORRUPTIONS = [
    'identity',
    'shot_noise',
    'impulse_noise',
    'glass_blur',
    'motion_blur',
    'shear',
    'scale',
    'rotate',
    'brightness',
    'translate',
    'fog',
    'stripe',
    'spatter',
    'dotted_line',
    'zigzag',
    'canny_edges',
]

_TEST_IMAGES_FILENAME = 'test_images.npy'
_TEST_LABELS_FILENAME = 'test_labels.npy'

# Change esp to change the probability of getting a corrupted image during testing.
esp = 0.5
number_of_features = 28*28
n_clauses = 8000*10
epochs = 100

(X_train, Y_train), (_, _) = mnist.load_data()
all_test_images = []
all_test_labels = []

for corruption in CORRUPTIONS:
    images_file = os.path.join(data_dir, corruption, _TEST_IMAGES_FILENAME)
    labels_file = os.path.join(data_dir, corruption, _TEST_LABELS_FILENAME)
    images = np.load(images_file)
    labels = np.load(labels_file)
    all_test_images.append(images)
    all_test_labels.append(labels)

all_test_images = np.array(all_test_images)
all_test_labels = np.array(all_test_labels)

print(all_test_images.shape, all_test_labels.shape)
print("___________________________________________")

X_test = []
Y_test = []

for i in range(10000):
    corrupt = np.random.choice(range(len(CORRUPTIONS)),p=[1-esp]+[esp/(len(CORRUPTIONS)-1)]*(len(CORRUPTIONS)-1))
    X_test.append(all_test_images[corrupt][i])
    Y_test.append(all_test_labels[corrupt][i])
X_test = np.array(X_test)
Y_test = np.array(Y_test)
print(X_test.shape, Y_test.shape)

#X_train = np.expand_dims(X_train, axis=-1)
#X_test = np.expand_dims(X_test, axis=-1)
X_train = np.where(X_train >= 75, 1, 0) 
X_test = np.where(X_test >= 150, 1, 0)
'''
for i in range(X_test.shape[0]):
                X_test[i,:,:] = cv2.adaptiveThreshold(X_test[i,:,:], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
for i in range(X_train.shape[0]):
                X_train[i,:,:] = cv2.adaptiveThreshold(X_train[i,:,:], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
'''
X_train = np.reshape(X_train,(X_train.shape[0],number_of_features))
X_test = np.reshape(X_test,(X_test.shape[0],number_of_features))

tm = MultiClassTsetlinMachine(n_clauses, 7000, 5.0, boost_true_positive_feedback=1)

#f = open("mnist_%.1f_%d_%d_%d_%.2f.txt" % (s, clauses, T,  patch_size, drop_clause), "w+")
max = 0.0

for i in range(epochs):
		tm.fit(X_train, Y_train, epochs=1, incremental=True)

		result_test = 100*(tm.predict(X_test) == Y_test).mean()
		result_train = 100*(tm.predict(X_train) == Y_train).mean()
        
		if result_test > max:
			max = result_test

		print("#%d Train Accuracy: %.2f,  Test Accuracy: %.2f\n" % (i, result_train, result_test))
		#print("#%d Accuracy: %.2f\n" % (i, result_test), file=f)
		#f.flush()
