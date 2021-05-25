# Human Interpretable AI: Randomized StochasticTsetlin Machine with Drop Clause

Install the drop clause multi-gpu Tsetlin machine:

	python setup.py install

	pip install requirements.txt

Using the drop clause TM:

	from PyTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D

	tm = MultiClassConvolutionalTsetlinMachine2D(clauses, T, s, (patch_size, patch_size), clause_drop_p = drop_clause, number_of_gpus=n_gpus, number_of_state_bits=number_of_state_bits)
	tm.fit(train_data, train_labels, epochs=1, incremental=True)
	accuracy = 100*(tm.predict(test_data) == test_labels).mean()


Running the code (from examples folder):

	With default parameters- python CIFARDemo2DConvTM_Interpret.py --gpus 1 --stop_train 1000

	With Interpretability- python CIFARDemo2DConvTM_Interpret.py --interpret True

	With changes in parameters python CIFARDemo2DConvTM_Interpret.py --n_clauses_per_class 30000 --s 10.0 --T 750 --drop_clause 0.5 --patch_size 8 --gpus 8

	Similarly for NLP Interpretability- python SSTDemoWeightedClauses_Interpret.py
