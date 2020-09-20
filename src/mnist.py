#!/usr/bin/env python3
import numpy as np
from nn import NeuralNetwork
import os, random, tqdm


def main():
	# Load the dataset
	test_set = np.loadtxt(os.path.abspath('mnist/mnist_test.csv'), delimiter=',')
	train_set = np.loadtxt(os.path.abspath('mnist/mnist_train.csv'), delimiter=',')




	nn = NeuralNetwork(784, 50, 50, 10,
					   lr=.01,
					   activation='tanh')

	train(nn, train_set)
	predict(test_set)

def train(nn, train_data, epoch=3):
	for e in range(epoch):
		ind_list = [i for i in range(len(train_data))]
		random.shuffle(ind_list)
		for i in tqdm.tqdm(ind_list,
						   ncols=100,
                  		   desc=f'Training! epoch ({e+1}/{epoch})'):
			target = [0 for x in range(10)]
			#print(train_data[i][0])
			target[int(train_data[i][0])] = 1
			data = [x / 255 for x in train_data[i][1:]]
			nn.train(data, target)
		#nn.lr/=10
	nn.save_model('mnist_test.P')

def predict(test_data):
	nn = NeuralNetwork()
	nn.load_model('mnist_test.P')
	accuracy = 0
	predictions = [0 for x in range(10)]
	for i in tqdm.tqdm(range(len(test_data)),
					   ncols=100,
                  	   desc='Testing!'):
		target = test_data[i][0]
		predict = nn.predict(test_data[i][1:])
		predict = list(predict)
		predict = predict.index(max(predict))
		if predict == target:
			accuracy += 1
		predictions[predict] += 1
	accuracy /= len(test_data)

	print(f'\n\n---------Results------------\n')
	print(f'Activation Function: {nn.activation.__name__}')
	#print(f'Learning Rate: {nn.lr}')
	print(f'Accuracy: {accuracy*100}%')
	print(f'Prediction Totals: {predictions}')

if __name__ == '__main__':
	main()
