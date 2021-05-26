from PyTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D
import numpy as np
from time import time
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import fashion_mnist
import random
import cv2
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-interpret', type=bool, default=False)
parser.add_argument('-n_clauses_per_class', type=int, default=8000)
parser.add_argument('-s', type=float, default=5.0)
parser.add_argument('-T', type=int, default=250)
parser.add_argument('-drop_clause', type=float, default=0.25)
parser.add_argument('-state_bits', type=int, default=12)
parser.add_argument('-patch_size', type=int, default=10)
parser.add_argument('-gpus', type=int, default=16)
parser.add_argument('-stop_train', type=int, default=250)

config = parser.parse_args()

factor = 40

s = config.s

T = int(factor*config.T)

clauses = config.n_clauses_per_class
n_classes = 10

drop_clause = config.drop_clause
n_gpus = config.gpus

ensembles = 1
epochs = config.stop_train

batches = 10

patch_size = config.patch_size
image_size = 28

number_of_x_pos_features = image_size - patch_size
number_of_y_pos_features = image_size - patch_size
number_of_content_features = patch_size*patch_size
number_of_features = number_of_x_pos_features + number_of_y_pos_features + number_of_content_features

number_of_state_bits = config.state_bits

def get_lower_upper_x_y_masks(class_id, clause):
    # Get lower bound for X and Y coordinates
    lower_y = -1
    lower_x = -1
    for k in range(number_of_x_pos_features):
        if tm.ta_action(class_id, clause, k) == 1:
            lower_y = k
        if tm.ta_action(class_id, clause, number_of_y_pos_features + k) == 1:
            lower_x = k

    # Get upper bound for X and Y coordinates
    upper_y = image_size
    upper_x = image_size
    for k in range(number_of_x_pos_features-1, -1, -1):
        if tm.ta_action(class_id, clause, number_of_features + k) == 1:
            upper_y = k
        if tm.ta_action(class_id, clause, number_of_features + number_of_y_pos_features + k) == 1:
            upper_x = k

    # If patch mask_1 contains 1, the correponding image pixel must also be 1.
    mask_1 = np.zeros((patch_size, patch_size)).astype(np.int8)
    for patch_y in range(patch_size):
        for patch_x in range(patch_size):
            feature = number_of_x_pos_features + number_of_y_pos_features + patch_y * patch_size + patch_x
            if tm.ta_action(class_id, clause, feature) == 1:
                mask_1[patch_x, patch_y] = 1

    # If patch mask_0 contains 1, the correponding image pixel must also be 0 (negated features)
    mask_0 = np.zeros((patch_size, patch_size)).astype(np.int8)
    for patch_y in range(patch_size):
        for patch_x in range(patch_size):
            feature = number_of_features + number_of_x_pos_features + number_of_y_pos_features + patch_y * patch_size + patch_x
            if tm.ta_action(class_id, clause, feature) == 1:
                mask_0[patch_x, patch_y] = 1
    
    return lower_x, lower_y, upper_x, upper_y, mask_1, mask_0

(X_train, Y_train), (X_test1, Y_test) = fashion_mnist.load_data()
X_train = np.copy(X_train)
X_test = np.copy(X_test1)

for i in range(X_train.shape[0]):
	X_train[i,:] = cv2.adaptiveThreshold(X_train[i], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

for i in range(X_test.shape[0]):
	X_test[i,:] = cv2.adaptiveThreshold(X_test[i], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

f = open("fmnist_%.1f_%d_%d_%d.txt" % (s, clauses, T,  patch_size), "w+")

for e in range(ensembles):
	tm = MultiClassConvolutionalTsetlinMachine2D(clauses*n_classes, T, s, (patch_size, patch_size), clause_drop_p = drop_clause, number_of_gpus = n_gpus, number_of_state_bits=number_of_state_bits)

	for i in range(epochs):
		start_training = time()
		tm.fit(X_train, Y_train, epochs=1, incremental=True)
		stop_training = time()

		start_testing = time()
		result_test = 100*(tm.predict(X_test) == Y_test).mean()
		stop_testing = time()

		result_train = 100*(tm.predict(X_train) == Y_train).mean()

		print("%d %d %.2f %.2f %.2f %.2f" % (e, i, result_train, result_test, stop_training-start_training, stop_testing-start_testing))
		print("%d %d %.2f %.2f %.2f %.2f" % (e, i, result_train, result_test, stop_training-start_training, stop_testing-start_testing), file=f)
		f.flush()

		if config.interpret:
        
			if max > 90.0:
            
				class_id = 0
				clause = 0
				max_weight = 0
                
				for i in range(9):
					for j in range(clauses//10):
						if tm.get_weight(i, j) > max_weight:
							class_id = i
							clause = j
							max_weight = tm.get_weight(i, j)

				print("Class: %d" %(class_id), file=f)
				print("Max_weight: %d" %(max_weight), file=f)

				clause_weights_for_class = []
				for k in range(clauses//10):
					clause_weights_for_class.append(tm.get_weight(class_id, k))

				weight_indices = sorted(range(len(clause_weights_for_class)), key=lambda k: clause_weights_for_class[k], reverse=True)
                
				Max_class_instances = []
				Max_class_instances_original = []
				for q in range(X_test.shape[0]):
					if Y_test[q] == class_id and tm.predict(np.expand_dims(X_test[q], axis=0)) == Y_test[q]:
						Max_class_instances.append(X_test[q])
						Max_class_instances_original.append(X_test1[q])
						break
                
				outputs = np.zeros((image_size,image_size)).astype(np.uint8)
                
                #Clauses with top 100 weights
                # Combined mask (-1 must be 0 and 1 must be 1 for the corresponding image pixel value, 0 means ignore image pixel value)

				np.set_printoptions(linewidth=np.inf)

				for p in range(100):
					output = np.zeros((image_size,image_size)).astype(np.uint8)
					lower_x, lower_y, upper_x, upper_y, mask_1, mask_0 = get_lower_upper_x_y_masks(class_id, weight_indices[p])
                    #print("Max Weight Number: ", p+1)
                    #print("Weight: ", clause_weights_for_class[weight_indices[p]])
                    #print(lower_x, "< x <=", upper_x)
                    #print(lower_y, "< y <=", upper_y)
					mask = mask_1 - mask_0
					for i in range(lower_x, upper_x):
						for j in range(lower_y, upper_y):
							for k in range(patch_size):
								for l in range(patch_size):
									if i+k >= image_size or j+l >= image_size:
										break
									elif mask[k][l] == 1 and Max_class_instances[0][i+k][j+l] == 1:
										output[i+k][j+l] = 1
									elif mask[k][l] == -1 and Max_class_instances[0][i+k][j+l] == 0:
										output[i+k][j+l] = 1                    
					outputs += output
                    
				plt.imshow(Max_class_instances_original[0].astype(np.uint8), interpolation='nearest')
				plt.savefig('img_dc_%.2f_%d.jpeg' %(drop_clause,batch))
				plt.imshow(outputs, cmap='hot', interpolation='nearest')
				plt.savefig('heatmap_dc_%.2f_%d.png' %(drop_clause,batch))
				plt.imshow(Max_class_instances_original[0], interpolation='nearest')
				plt.imshow(outputs, cmap='cool', interpolation='nearest', alpha=0.5)
				plt.savefig('img_heatmap_dc_%.2f_%d.png' %(drop_clause,batch))
            
f.close()
