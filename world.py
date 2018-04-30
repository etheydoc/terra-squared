# SeraphimNashville:Science:AI:ProjectTerra^2//fractal AI + cellular automata
# Class that defines and handles the simulated world and weather
# (c) Seraphim 2018

# Created by Ethan Block
# April 25, 2018

import proc
import numpy as np

class TSQWorld:

    def __init__(self, size=32):
        self.size = (size, size)
        self.terrain = proc.gen_terrain(size)
        self.terrain /= np.max(self.terrain)
        self.climates = proc.gen_climates(size)

    def update_vegetation(self):
        pass

    def update_climate(self):
        pass

    def update_storms(self):
        pass

    def update(self):
        pass
