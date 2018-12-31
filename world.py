# SeraphimNashville:Science:AI:ProjectTerra^2//fractal AI + cellular automata
# Class that defines and handles the simulated world and weather
# (c) Seraphim 2018

# Created by Ethan Block
# April 25, 2018

import proc, random
import numpy as np

class TSQWorld:

    def __init__(self, size=32, wlev=0.5):
        """Initialize the Terra^2 world.
        :size: the size of the map (in units, must be 2^n)
        :wlev: the water level of the world (0-1)
        """
        self.size = (size, size)
        self.wlev = wlev
        self.terrain = proc.gen_terrain(size, seed=random.randint(0,999999))
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

    def find_spawn(self):
        possible = np.array(np.where(self.terrain > self.wlev)).T
        return random.choice(possible)
