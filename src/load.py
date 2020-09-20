import pickle
import numpy as np

with open ('test.model', 'rb') as fp:
    weights = pickle.load(fp)

for i, w in enumerate(weights):
	print(f'{i+1}: {w}')

