from bird import Bird
from nn import NeuralNetwork
import random
from numpy import argmax

#1. create a population
#2. calculate fitness
#3. choose 2 from selection pool
#4. crossover dna
#5. mutate
#6. return new population

class evolution(object):
    def __init__(self, popsize, surf, mr=.01):
        self.surf = surf
        self.popsize = popsize
        self.mr = mr
        self.population = [] 
        self.fit_arr = []
        self.deadcount = 0
        self.init_pop()
        self.num_gens = 0


    def init_pop(self):
        for _ in range(self.popsize):
            nn = NeuralNetwork(5,8,1, activation='relu')
            self.population.append(Bird(self.surf, (0, 0), 20, max_vel=20, DNA=nn))
        

    def reset(self):
        for obj in self.population:
            obj.reset()
            obj.DNA.reset()


    def calc_fitness(self):
        del self.fit_arr[:]
        s = 0
        for obj in self.population:
            s += obj.fitness
        for obj in self.population:
            obj.fitness/=s
            self.fit_arr.append(obj.fitness)

         
    def pick_one(self):
        index = 0
        r = random.random()
        while (r > 0):
            r = r - self.population[index].fitness
            index += 1
        index -= 1
        return self.population[index]


    def crossover(self):
        a = self.pick_one()        
        b = self.pick_one()
        w, b = a.DNA.crossover(b.DNA)
        childDNA = NeuralNetwork(5, 8, 1, weights=w, bias=b)
        childDNA.mutate(self.mr)
        return childDNA


    def next_gen(self):
        new_pop = []
        for _ in range(self.popsize):
            new_pop.append(Bird(self.surf, (0, 0), 20, max_vel=20, DNA=self.crossover()))
        self.population = new_pop


    def evaluate(self, pipes):
        if self.deadcount >= self.popsize:
            self.deadcount = 0
            self.calc_fitness()
            self.next_gen()
            self.num_gens += 1 
            for pipe in pipes:
                pipe.reset()


    def run(self, pipes):
        for obj in self.population:
            if not obj.dead:
                obj.think(pipes)
                obj.update()
                obj.collision(self.surf, pipes)
                if obj.dead:
                    self.deadcount+=1
        for pipe in pipes:
            pipe.update()
        self.evaluate(pipes)


    def show(self, pipes):
        for obj in self.population:
            if not obj.dead:
                obj.show(self.surf, pipes, lines=False)


    def save_best(self, fn):
        self.calc_fitness()
        index = argmax(self.fit_arr)         
        self.population[index].DNA.save_model(fn)











