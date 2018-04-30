# SeraphimNashville:Science:AI:ProjectTerra^2//fractal AI + cellular automata
# Class that defines and handles Terran populations (simulated fractal AI swarms)
# (c) Seraphim 2018

# Created by Ethan Block
# April 25, 2018

import util

class Terran(util.Struct):
    energy = 1.
    health = 1.
    social = 1.

class TerranPop:

    def __init__(self, num_terrans):
        self.terrans = [Terran() for i in range(num_terrans)]

    def update(self):
        pass
