# -*- coding: utf-8 -*-
"""
pso.py

Test functions: https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""

import random
import math

params = dict(
    swarm_size=75,
    velocity_retention=0.5,
    pb_retention=0.5,
    ib_retention=0.5,
    gb_retention=0.5,
    jump_size=0.1,
    iterations=500
)


# Fitness functions

def sphere(position):
    return sum([x ** 2 for x in position])

def rosenbrock(position):    
    return sum([100 * (position[i+1] - position[i] ** 2) ** 2 + (position[i] - 1) ** 2 for i in range(len(position) - 1)])

def rastrigin(position):
    A = 10
    return A * len(position) + sum([x ** 2 - A * math.cos(2 * math.pi * x) for x in position])

# Setting fitness function to be used
fitness = rastrigin

def fittest(p1, p2):
    f1 = fitness(p1)
    f2 = fitness(p2)
    return p1 if f1 < f2 else p2


def new_vector(size):
    # TODO: Refactor to consider the apropriate search domain for each objective function
    return [random.uniform(-5.12, 5.12) for d in range(size)]


class Particle:
    def __init__(self, size=2):
        self.position = new_vector(size)
        self.velocity = new_vector(size)
        self.best = self.position

    def update(self, gb):
        vr = random.uniform(0, params['velocity_retention'])
        pbr = random.uniform(0, params['pb_retention'])
        gbr = random.uniform(0, params['gb_retention'])
        jump = params['jump_size']
        # TODO: Refactor this loop
        for i in range(len(self.velocity)):
            self.velocity[i] = vr * self.velocity[i] + pbr * (self.best[i] - self.position[i]) + gbr * (gb[i] - self.position[i])
        for j in range(len(self.position)):
            self.position[j] = self.position[j] + jump * self.velocity[j]

    def __repr__(self):
        return 'Particle(position={0}, velocity={1})'.format(self.position, self.velocity)

# TODO: Study how the multprocessing module could make futere multiple swarm implementation more efficient
# See: https://docs.python.org/3/library/multiprocessing.html
class Swarm:
    def __init__(self, size=100):
        self.population = [Particle() for i in range(size)]
        self.best = self[0].position

    def _update_best(self):
        for p in self.population:
            self.best = fittest(self.best, p.position)

    def update(self):
        for p in self:
            p.update(self.best)
        self._update_best()

    def __len__(self):
        return len(self.population)

    def __getitem__(self, key):
        return self.population[key]

    def __setitem__(self, key, value):
        self.population[key] = value

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.population):
            i = self.index
            self.index += 1
            return self.population[i]
        raise StopIteration

    def __repr__(self):
        return 'Swarm(size={0}, population={1})'.format(len(self), self.population)


class PSO():
    def __init__(self, *args, **kwargs):
        self.swarm_size = kwargs['swarm_size']
        self.velocity_retention = kwargs['velocity_retention']
        self.pb_retention = kwargs['pb_retention']
        self.ib_retention = kwargs['ib_retention']
        self.gb_retention = kwargs['gb_retention']
        self.jump_size = kwargs['jump_size']
        self.iterations = kwargs['iterations']

    def run(self):
        swarm = Swarm(self.swarm_size)
        for i in range(1, self.iterations + 1):
            swarm.update()
            if i % int(self.iterations / 20) == 0:
                print("Iteration {:4d}: best = {} (fitness = {})".format(i, swarm.best, fitness(swarm.best)))


def main():
    pso = PSO(**params)
    pso.run()


if __name__ == '__main__':
    main()
