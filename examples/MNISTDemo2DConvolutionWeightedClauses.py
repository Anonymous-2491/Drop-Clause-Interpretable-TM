from PyTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D

import numpy as np
from time import time

from keras.datasets import mnist

factor = 40

s = 5.0

T = int(factor*25*10)

clauses = int(factor*2000)

ensembles = 10
epochs = 250

batches = 10

patch_size = 10
image_size = 28

number_of_x_pos_features = image_size - patch_size
number_of_y_pos_features = image_size - patch_size
number_of_content_features = patch_size*patch_size
number_of_features = number_of_x_pos_features + number_of_y_pos_features + number_of_content_features

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train >= 75, 1, 0) 
X_test = np.where(X_test >= 75, 1, 0)

f = open("mnist_%.1f_%d_%d_%d.txt" % (s, clauses, T,  patch_size), "w+")

for e in range(ensembles):
	tm = MultiClassConvolutionalTsetlinMachine2D(clauses, T, s, (patch_size, patch_size), clause_drop_p = 0.01, feature_drop_p = 0.01, number_of_gpus = 16)

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

		# Example of getting a single clause (find the one with the largest weight)
		class_id = 0
		clause = 0
		max_weight = 0
		for i in range(10):
			for j in range(clauses//10): 
				if tm.get_weight(i, j) > max_weight:
					class_id = i
					clause = j
					max_weight = tm.get_weight(i, j)

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

		# Combined mask (-1 must be 0 and 1 must be 1 for the corresponding image pixel value, 0 means ignore image pixel value)
		print("Weight:", max_weight)

		print(lower_x, "< x <=", upper_x)
		print(lower_y, "< y <=", upper_y)

		print(mask_1 - mask_0)

		# Example of getting a single clause (find the one with the largest weight)
		class_id = 0
		clause = 0
		min_weight = 0
		for i in range(10):
			for j in range(clauses//10): 
				if tm.get_weight(i, j) < min_weight:
					class_id = i
					clause = j
					min_weight = tm.get_weight(i, j)

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

		# Combined mask (-1 must be 0 and 1 must be 1 for the corresponding image pixel value, 0 means ignore image pixel value)
		print("Weight:", min_weight)
		print(lower_x, "< x <=", upper_x)
		print(lower_y, "< y <=", upper_y)
		print(mask_1 - mask_0)
f.close()
