import pygame
from pygame.locals import *
from pygame.math import *
import random
from copy import deepcopy
from nn import NeuralNetwork


class Bird(object):
    def __init__(self, surf, vel=(0,0), r=20, max_vel=.8, DNA=None):
        self.surf = surf
        w, h = self.surf.get_size()
        self.initial_pos = Vector2(w//3, h//3)
        self.initial_vel = Vector2(vel)
        self.pos = deepcopy(self.initial_pos)
        self.vel = deepcopy(self.initial_vel)
        self.max_vel = max_vel
        self.r = r
        self.DNA = DNA
        self.dead = False
        self.fitness = 0


    def load(self, fn):
        nn = NeuralNetwork(5, 8, 1)
        nn.load_model(fn)
        self.DNA = nn


    def update(self):
        self.fitness+=1
        g = Vector2(0,.7)
        self.vel += g
        if self.vel.y > self.max_vel:
            self.vel.y = self.max_vel
        elif self.vel.y < -self.max_vel*3/5:
            self.vel.y = -self.max_vel*3/5
        self.pos += self.vel


    def think(self, pipes):
            #get closest pipe to the right
            w, h = self.surf.get_size()
            m = 100000
            index = 0
            for i, pipe in enumerate(pipes):
                pipe.closest = False
                dist = pipe.pos.x + pipe.w - self.pos.x + self.r
                if dist < m and not dist < 0:
                    m = dist
                    index = i
            pipes[index].closest = True
            t, b = pipes[index].get_dim()
            dx = (self.pos.x - t.x)/w*10
            dy_t = (self.pos.y - t.y)/h*10
            dy_b = (self.pos.y - b.y)/h*10
            inputs = [self.pos.y/h, self.vel.y/self.max_vel, dx, dy_t, dy_b]
            try:
                output = self.DNA.predict(inputs)
                if output > .5:
                    self.apply_force((0,-self.max_vel))
            except AttributeError:
                print('no dna')
            return inputs


    def reset(self):
        self.pos = deepcopy(self.initial_pos)
        self.vel = deepcopy(self.initial_vel)
        self.fitness = 0
        self.dead = False


    def collision(self, surf, pipes):
        if self.edges(surf):
            self.dead = True
        else:
            for pipe in pipes:
                x = self.pos.x + self.r
                y = self.pos.y - self.r
                pos = Vector2(x, y)
                t, b = pipe.get_dim()
                #top
                if pos.x > t.x and pos.x < t.x + pipe.w and pos.y < t.y:
                    self.dead = True
                #bottom
                elif pos.x < b.x + pipe.w and pos.x > b.x and pos.y + self.r*2 > b.y:
                    self.dead = True


    def edges(self, surf):
        w, h = surf.get_size()
        if self.pos.x < -self.r/2:
            return True
        elif self.pos.x > w + self.r/2:
            return True
        elif self.pos.y + self.r/2 > h:
            return True
        elif self.pos.y + self.r/2 < 0:
            return True
        else:
            return False


    def apply_force(self, f):
        self.vel += Vector2(f)


    def show(self, surf, pipes, lines=False):
        if lines:
            w, h = self.surf.get_size()
            m = 100000
            index = 0
            for i, pipe in enumerate(pipes):
                pipe.closest = False
                dist = pipe.pos.x + pipe.w - self.pos.x + self.r
                if dist < 0 and not pipe.passed:
                    pipe.passed = True
                if dist < m and not dist < 0:
                    m = dist
                    index = i
            pipes[index].closest = True
            t, b = pipes[index].get_dim()

            pygame.draw.line(surf, (255,255,255),
                            (self.pos.x, self.pos.y),
                            (t.x, t.y))
            pygame.draw.line(surf, (255,255,255),
                            (self.pos.x, self.pos.y),
                            (b.x, b.y))
        pygame.draw.circle(surf, (255,0,0),
                         (int(self.pos.x), int(self.pos.y)),
                         self.r, 0)


class Pipe(object):
    def __init__(self, surf, w, gap, speed, offset=0):
        self.offset = offset
        self.w = w
        self.gap = gap
        self.surf = surf
        self.surfx, self.surfy = self.surf.get_size()
        self.l = random.randint(self.surfy//8, self.surfy - self.surfy//8*2)
        self.initial_pos = Vector2(self.surfx+self.offset ,0)
        self.pos = deepcopy(self.initial_pos)        
        self.vel = Vector2(-speed, 0)
        self.closest = False
        self.passed = False


    def reset(self):
        self.pos = deepcopy(self.initial_pos)
        self.closest = False
        self.l = random.randint(self.surfy//8, self.surfy - self.surfy//8*2)


    def show(self):
        col = (0,100,0)
        #top half
        pygame.draw.rect(self.surf, col,
                         pygame.Rect(self.pos.x, self.pos.y, self.w, self.l))
        #bottom half
        pygame.draw.rect(self.surf, col,
                         pygame.Rect(self.pos.x, self.l + self.gap, self.w,
                                     self.surfy))


    def update(self):
        self.pos += self.vel
        if self.pos.x + self.w < 0:
            self.pos.x = self.surfx + self.w
            self.l = random.randint(self.surfy//8, self.surfy - self.surfy//8*2)
           # self.l = random.randint(100, self.surfy - 200)
            self.passed = False


    def get_dim(self):
        top = Vector2(self.pos.x, self.l)
        bot = Vector2(self.pos.x, self.l + self.gap)
        return top, bot
