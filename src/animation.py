#!/usr/bin/python3
import numpy as np
import time
import random
import pygame
import sys
from pygame.locals import *
from nn import NeuralNetwork

class Node(object):
    def __init__(self, x, y, col, r):
        self.x = x
        self.y = y
        self.r = r
        self.col = pygame.Color(0, 0, 0)
        self.col.hsva = (col, 100, 100, 100)
        self.w_cols = []        
        self.bias_col = self.col 

    def show(self, surf, fill):
        pygame.draw.circle(surf, self.col,
                          (self.x, self.y), self.r, fill) 
        pygame.draw.circle(surf, self.bias_col,
                          (self.x, self.y), self.r//2, fill) 

    #makes it look kinda confusing    
    def add_bias(self, bias):
        self.bias_col = pygame.Color(0,0,0)
        col = constrain(remap(bias, -2, 2, 0, 120), 0, 120)
        if col >= 30 and col <= 90:
            self.bias_col = self.col
        else:
            self.bias_col.hsva = (col, 100,100,100)


    def add_weight(self, col):
        self.w_cols.append(pygame.Color(0,0,0))         
        col = constrain(col, 0, 120)        
        bright = abs(col - 60)
        bright = constrain(bright, 0, 100)
        if self.col.hsva[0] < 60:
            bright = 0
        self.w_cols[-1].hsva = (col, 100, bright, 100)

def pause():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                 pygame.quit()
                 sys.exit() 
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    return
                if event.key == K_ESCAPE:
                    return
                    
def animate_training(nn, training_data):
    pygame.init()
    display = pygame.display.set_mode((800,600))
    pygame.display.set_caption('a neural net hopefully')
    w, h = pygame.display.get_surface().get_size()
    
    delay = 1
    delay_count = 0
    count = 0
    while True:
        display.fill((40,40,40))
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit
                return
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    return
                if event.key == K_DOWN:
                    delay += 100
                    delay_count +=10
                    if delay > 1000:
                        delay = 1000
                        delay_count += 1000
                if event.key == K_UP:
                    delay -= 100
                    if delay <= 0:
                        delay = 1
                if event.key == K_RIGHT:
                    delay = 1
                if event.key == K_LEFT:
                    delay = 1000
                    delay_count +=1000
                if event.key == K_SPACE:
                    pause() 
        if delay_count % delay == 0:
            data = random.choice(training_data)
            targets = (data[0])
            inputs = (data[1:])
            outputs, weights, bias, rms = nn.train(inputs,targets)
            #build nodes
            layers = []
            for n in nn.layers:
                layers.append([0]*n)
            for x, o in enumerate(outputs):
                num_nodes = o.shape[0]
                num_layers = len(nn.layers)
                for y in range(num_nodes):
                    col = constrain(120*o[y], 0, 120)
                    x_pos = calc_pos(num_layers, x, w)
                    y_pos = calc_pos(num_nodes, y, h)
                    n = Node(x_pos, y_pos, col, int(50/(np.sqrt(len(o)))))
                    layers[x][y] = n
            count+=1
        #draw nodes and weights
        for i, layer in enumerate(layers):
            for j, node in enumerate(layer):
                if i < len(weights): 
                    for k in range(len(weights[i])):
                        if nn.activation.__name__ == 'sigmoid':
                            col = remap(weights[i][k][j], -2, 2, 0, 120)
                        else:
                            col = remap(weights[i][k][j], -1, 1, 0, 120)
                        n1 = layers[i][j]
                        n1.add_weight(col)
                        n2 = layers[i+1][k]
                        pygame.draw.line(display, n1.w_cols[k],
                                        (n1.x, n1.y), (n2.x, n2.y))
                #makes things confusing
                # if i != 0:
                #     n1.add_bias(bias[i-1][j])
        for layer in layers:
            for node in layer:
                node.show(display, 0)
        message_display('Learning rate: {}%'.format(nn.lr), display, w+30, 30)
        message_display('AF: {}'.format(nn.activation.__name__), display, w+30, 55)
        message_display('Iterations: {}'.format(count), display, w+30, 80)
        message_display('Error = {:03.04f}% '.format(float(rms)*100), display, w+30, 105)

        delay_count += 1 
        pygame.display.flip()


def animate_predict(nn, input_data, surf):
    display = surf
    w, h = pygame.display.get_surface().get_size()
    inputs = input_data
    outputs, weights, bias = nn.feed_forward(inputs)

    #build nodes
    layers = []
    for n in nn.layers:
        layers.append([0]*n)
    for x, o in enumerate(outputs):
        num_nodes = o.shape[0]
        num_layers = len(nn.layers)
        for y in range(num_nodes):
            col = constrain(120*o[y]*1.2, 0, 120)
            x_pos = calc_pos(num_layers, x, w//2)
            x_pos += w//2
            y_pos = calc_pos(num_nodes, y, h//2)
            n = Node(x_pos, y_pos, col, int(50/(np.sqrt(len(o)))))
            layers[x][y] = n
        
    #draw weights
    for i, layer in enumerate(layers):
        for j, node in enumerate(layer):
            if i < len(weights): 
                for k in range(len(weights[i])):
                    if nn.activation.__name__ == 'sigmoid':
                        col = remap(weights[i][k][j], -1, 1, 0, 120)*1.4
                    else:
                        col = remap(weights[i][k][j], -1, 1, 0, 120)
                    n1 = layers[i][j]
                    n1.add_weight(col)
                    n2 = layers[i+1][k]
                    pygame.draw.line(display, n1.w_cols[k],
                                    (n1.x, n1.y), (n2.x, n2.y))
    #draw nodes
    for layer in layers:
        for node in layer:
            node.show(display, 0)

#add these to util
def calc_pos(items, n, dim):
    pos = dim//(items+1)
    pos += n*pos
    return pos 

def text_objects(text, font):
    textSurface = font.render(text, True, (255,255,255))
    return textSurface, textSurface.get_rect()

def message_display(text, display, w, h):
    largeText = pygame.font.Font('freesansbold.ttf',20)
    TextSurf, TextRect = text_objects(text, largeText)
    TextRect.center = ((w-150),h)
    display.blit(TextSurf, TextRect)

def constrain(x, _min, _max):
    return max(min(_max, x), _min)

def remap(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)
    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

if __name__ == '__main__':
    nn = NeuralNetwork(2,16,16,1,
                       activation='sigmoid',
                       lr=.1)
    training_data = [[1,1,0], [1,0,1], [0,1,1], [0,0,0]]
    animate_training(nn, training_data)
