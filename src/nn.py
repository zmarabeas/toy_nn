import numpy as np
from util.functions import *
import pickle
import random
from copy import deepcopy


VALID_KEYS = ['lr', 'cost', 'activation', 'weights', 'bias']
ACTIVATION_FUNCTIONS = {
    'sigmoid': sigmoid,
    'tanh': tanh,
    'relu': relu,
    'leaky_relu': leaky_relu,
}


class NeuralNetwork(object):
    def __init__(self, *args, **kwargs):
        self.layers = []
        self.weights = []
        self.lr = 0.01
        self.bias = []
        self.cost = None
        self.activation = relu

        #make a function to do this / deal with activation functions
        for arg in args:
            self.layers.append(arg)        
        for key, val in kwargs.items():
            if key in VALID_KEYS:
                setattr(self, key, val)
            else:
                print('In NeuralNetwork: invalid keyword')
        try:
            self.activation = ACTIVATION_FUNCTIONS[self.activation]
        except KeyError:
            print('Invalid activation function. Defaulting to relu')
            self.activation = ACTIVATION_FUNCTIONS['relu']

        if not self.weights and not self.bias:
            self.weights, self.bias = self._init_weights()
            

    def _init_weights(self):
        weights = []
        bias = []
        for i in range(len(self.layers)-1):
           #getting numbers above 1 and below -1 for some reason
           weights.append(np.random.normal(0.0, pow(self.layers[i], -0.5), 
                         (self.layers[i+1], self.layers[i])))
           bias.append(np.random.normal(0.0, pow(self.layers[i], -0.5), 
                         (self.layers[i+1], 1)))
            #make this better
            # weights.append(np.random.random((self.layers[i+1], self.layers[i]))*2-1)
            # bias.append(np.random.random((self.layers[i+1], 1))*2-1)
        return weights, bias
    

    def train(self, input_list, input_target):
        target = np.array(input_target, ndmin = 2).T        
        assert target.size == self.layers[-1] #output layer
        #feedforward
        #list to hold outputs during feedforward
        outputs, _, _ = self.feed_forward(input_list) 
        out_copy = deepcopy(outputs)#activation funs are inplace
        #backpropogation
        #starting from right most layer
        #delta(jk) k == output layer to start  j == previous layer      
        #delta = lr * np.dot(error(k), (output(k)*(1 - output(k))*transpose(output(j)))
        err = target-outputs[-1]
        rms = np.sqrt((target-outputs[-1])**2)
        for i in range(len(self.layers)-1,0,-1):
            #calculate gradient, derivative of activation function *sigmoid
            #add to functions?? this is only for sigmoid
            #gradient = outputs[i]*(1-outputs[i])
            gradient = self.activation(outputs[i], True) 
            gradient = self.lr * (err * gradient)
            delta = np.dot(gradient,np.transpose(outputs[i-1])) 
            #update error
            err =  np.dot(np.transpose(self.weights[i-1]),err)
            #update weights and bias
            self.weights[i-1] += delta
            self.bias[i-1] += gradient
        return out_copy, self.weights, self.bias, rms


    #weights and bias are for animation
    def feed_forward(self, input_list):
        outputs = []
        data = np.array(input_list, ndmin=2).T
        outputs.append(data) 
        for w, b in zip(self.weights, self.bias):            
            data = self.activation(np.dot(w, data)+b)
            outputs.append(data)                        
        return outputs, self.weights, self.bias


    #returns only the final output
    def predict(self, input_list):
        data = np.array(input_list, ndmin=2).T
        for w, b in zip(self.weights, self.bias):            
            data = self.activation(np.dot(w, data)+b)
        return data 


    def save_model(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.weights, f)
            pickle.dump(self.bias, f)
            pickle.dump(self.activation, f)


    def load_model(self, file_name):
        with open (file_name, 'rb') as f:
            self.weights = pickle.load(f)
            self.bias = pickle.load(f)
            self.activation = pickle.load(f)
    

    def mutate(self, mr):
        for i in range(len(self.layers)-1):
            r = random.random()
            if r < mr:
                self.weights[i] = (np.random.normal(0.0, pow(self.layers[i], -0.5), 
                             (self.layers[i+1], self.layers[i])))
            r = random.random()
            if r < mr:
                self.bias[i] = (np.random.normal(0.0, pow(self.layers[i], -0.5), 
                             (self.layers[i+1], 1)))
            

    def crossover(self, other):
        split = random.randint(0,len(self.weights))        
        w = self.weights[:split] + other.weights[split:]
        split = random.randint(0,len(self.weights))        
        b = self.bias[:split] + other.bias[split:]
        return w, b
        

    def reset(self):
        self._init_weights()


    



