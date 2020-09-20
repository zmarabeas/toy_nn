#!/usr/bin/python3
from __future__ import division
import pygame
from random import randint
from math import fabs

PHYS_FPS = 30
DT = 1 / PHYS_FPS
MAX_FRAMETIME = 0.25



def interpolate(star1, star2, alpha):
    x1 = star1[0]
    x2 = star2[0]
    # since I "teleport" stars at the end of the screen, I need to ignore
    # interpolation in such cases. try 1000 instead of 100 and see what happens
    if fabs(x2 - x1) < 100:
        return (x2 * alpha + x1 * (1 - alpha), star1[1], star1[2])
    return star2


def run_game():
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((1500, 500))

    # generate stars
    stars = [(randint(0, 1500), randint(0, 1500), randint(2, 6)) for i in range(50)]
    stars_prev = stars

    accumulator = 0
    frametime = clock.tick()

    play = True
    while play:
        print(clock.get_fps())
        frametime = clock.tick() / 1000
        if frametime > MAX_FRAMETIME:
            frametime = MAX_FRAMETIME

        accumulator += frametime

        # handle events to quit on 'X' and escape key
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                play = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    play = False

        while accumulator >= DT:
            stars_prev = stars[:]
            # move stars
            for i, (x, y, r) in enumerate(stars):
                stars[i] = (x - r * 50 * DT, y, r) if x > -20 else (520, randint(0, 500), r)
            accumulator -= DT

        alpha = accumulator / DT
        stars_inter = [interpolate(s1, s2, alpha) for s1, s2 in zip(stars_prev, stars)]

        # clear screen
        screen.fill(pygame.Color('black'))

        # draw stars
        for x, y, r in stars_inter:
            pygame.draw.circle(screen, pygame.Color('white'), (int(x), y), r)

        pygame.display.update()

if __name__ == "__main__":
    run_game()
