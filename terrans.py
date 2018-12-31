# SeraphimNashville:Science:AI:ProjectTerra^2//fractal AI + cellular automata
# Class that defines and handles Terran populations (simulated fractal AI swarms)
# (c) Seraphim 2018

# Created by Ethan Block
# April 25, 2018

import util, random

class Terran(util.Struct):
    x = 0
    y = 0
    energy = 1.
    health = 1.
    social = 1.

class TerranPop:

    def __init__(self, num_terrans, world, pos=(0,0), bounds=(0,0)):
        self.terrans = [Terran(x=pos[0], y=pos[1]) for i in range(num_terrans)]
        self.bounds = bounds
        self.world = world

    def update(self):
        # temp - debug
        t = random.choice(self.terrans)
        newpos = (t.x + random.randint(-1,1), t.y + random.randint(-1,1))
        newpos = (max(min(newpos[0], self.world.terrain.shape[0]-1), 0), max(min(newpos[1], self.world.terrain.shape[1]-1), 0))
        if self.world.terrain[int(newpos[0]), int(newpos[1])] > self.world.wlev:
            t.x, t.y = newpos
