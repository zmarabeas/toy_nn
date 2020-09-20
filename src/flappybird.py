#!/usr/bin/python3

import pygame
from pygame.locals import *
from bird import Bird, Pipe
import random
from nn import NeuralNetwork
from ga import evolution 
from animation import *
import multiprocessing

RESOLUTION = (1600,800)
PHYS_FPS = 60
DT = 1 / PHYS_FPS
MAX_FRAMETIME = 0.25

PIPE_WIDTH = 75
PIPE_GAP = 175
PIPE_SPEED = 10

def run():
    pygame.init()
    screen = pygame.display.set_mode(RESOLUTION)
    pygame.display.set_caption('flap')
    w, h = screen.get_size()
    clock = pygame.time.Clock()
    x = w//2
    y = h//2

    accumulator = 0
    frametime = clock.tick()
    bird = Bird(screen, (0, 0), 20, max_vel=20)
    bird.load('example.P')
       
    pipes = []
    for i in range(0, w, w//2 + PIPE_WIDTH):
        pipes.append(Pipe(screen,PIPE_WIDTH, PIPE_GAP, PIPE_SPEED, offset = i))
    ga = evolution(500, screen)
    
    training = False
    done = False
    inputs = np.zeros((1,5)) 
    while not done:
        screen.fill((0,0,0))
        keys = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == K_SPACE:
                    bird.apply_force((0,-bird.max_vel))
                    pass
                if event.key == K_s:
                    ga.save_best('best_bird.P')
                if event.key == K_l:
                    try:
                        bird.load('best_bird.P')
                    except FileNotFoundError:
                        print('best_bird.P not found, loading example')
                        bird.load('example.P')
                    for pipe in pipes:
                        pipe.reset()
                    training = False
                if event.key == K_t:
                    for pipe in pipes:
                        pipe.reset()
                    ga = evolution(500, screen)
                    training = True
        if keys[pygame.K_ESCAPE]:
            done = True
        #fix for stutter
        frametime = clock.tick() / 1000
        if frametime > MAX_FRAMETIME:
            frametime = MAX_FRAMETIME
        accumulator += frametime
        while accumulator >= DT:
            if training:
                for _ in range(1):
                    ga.run(pipes)
            else:
                inputs = bird.think(pipes)
                bird.update()
                bird.collision(screen, pipes)
                for pipe in pipes:
                    pipe.update()
                if bird.dead:
                    bird.reset()
                    for pipe in pipes:
                        pipe.reset() 
            accumulator -= DT
        if training:
            ga.show(pipes)
            message_display(f'Generation {ga.num_gens}', screen,  w//2, 50)
        else:            
            animate_predict(bird.DNA,inputs, screen)
            bird.show(screen, pipes, lines=False)
        for pipe in pipes:
            pipe.show()
        
        pygame.display.flip()


def text_objects(text, font):
    textSurface = font.render(text, True, (200,200,200))
    return textSurface, textSurface.get_rect()


def message_display(text, display, w, h):
    largeText = pygame.font.Font('freesansbold.ttf',20)
    TextSurf, TextRect = text_objects(text, largeText)
    TextRect.center = ((w-150),h)
    display.blit(TextSurf, TextRect)


if __name__ == '__main__':
    run()
