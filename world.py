# Definition of terrain map (and its many layers) and procedural generation algo
# Ethan Block, 9-3-2018

import numpy as np
import scipy as sp
import scipy.ndimage

from numba import jit
import random
import util

@jit
def proc_gen(size, points, sigma):
    """Generate a square map of smoothed random points.

       Keyword arguments:
       size -- the size of the map (final map will be size squared)
       points -- the number of maximum points on the map. The higher this is, the higher the average value will be.
       sigma -- factor that determines how much the points are smoothed.
       """
    generated = np.zeros((size, size))
    gpoints = np.random.choice(range(0, size * size), size=int(points))
    for i in gpoints:
        generated[int(i / size), int(i % size)] = 1.0
    generated = proc_smooth(generated, sigma)
    generated = util.normalize(generated)
    return generated

@jit
def proc_smooth(map1, sigma):
    generated = sp.ndimage.filters.gaussian_filter(map1, [sigma, sigma], mode='constant')
    return generated

@jit
def proc_filter(map1, map2, threshold):
    if type(threshold) == tuple:
        map1[np.where(map2 < threshold[0])] = 0.0
        map1[np.where(map2 > threshold[1])] = 0.0
    else:
        map1[np.where(map2 <= threshold)] = 0.0
    return map1

class Terrain:

    def __init__(self, size, points=None, sigma=4, num_climates=10, v_sparsity=0.02, v_bounds=(0.2, 0.8), water_level=0.5,
                 s_rate=0.05):
        """A terrain object, which contains all information about the simulated world.

           Keyword arguments:
           size -- the size of the terrain.
           points -- the number of maximum points on the map in procedural generation.
           sigma -- smoothing factor for procedural generation.
           num_climates -- the number of climates in the terrain.
           v_sparsity -- the vegetation sparsity (as percent of the total map space).
           v_bounds -- the minimum and maximum climate values (temperatures) that vegetation can grow in.
           water_level -- the level of water on the map.
           s_rate -- the rate at which consumed vegetation (sustenance) is regrown.
           """

        self.size = size
        self.sigma = sigma
        self.v_sparsity = v_sparsity
        self.v_bounds = v_bounds
        self.water_level = water_level
        self.s_rate = s_rate

        if points is None:
            points = random.randint(size*10, size*20)

        self.points = points

        # generate the heightmap
        heightmap = proc_gen(size, points, sigma)
        self.heightmap = heightmap

        climates = proc_gen(size, num_climates, sigma)*4

        # here we set the polar regions to freezing
        climates[0:int(size/4)] = 0.0
        climates[-int(size/4):-1] = 0.0
        climates = proc_smooth(climates, sigma*4)
        climates = util.normalize(climates, bounds=(v_bounds[0]*0.75, v_bounds[1]*1.25))
        self.climates = climates
        self.gradient_c = util.get_gradient(climates)

        # spawn vegetation seeds across the map, but not in water or unsuitable climates

        vegetation = np.zeros((size, size))

        for i in range(int(points * v_sparsity)):
            veg_seed = (0, 0)

            while (climates[veg_seed] < v_bounds[0] or
                   climates[veg_seed] > v_bounds[1] or
                   heightmap[veg_seed] < water_level):
                   veg_seed = (random.randint(0, size-1), random.randint(0, size-1))

            vegetation[veg_seed] = 1.0

        vegetation = proc_smooth(vegetation, sigma/2)
        self.vegetation = proc_filter(vegetation, heightmap, water_level)
        self.sustenance = np.copy(vegetation) # we want changes to the vegetation to affect sustenance map

    def update(self):
        """Master update function for the terrain map."""
        self.grow_vegetation()

    def grow_vegetation(self):
        """Grow vegetation on the terrain map."""
        new_vegetation = proc_gen(self.size, int(self.points*self.v_sparsity), 0)
        new_vegetation = proc_filter(new_vegetation, self.climates, self.v_bounds)
        new_vegetation = proc_filter(new_vegetation, self.heightmap, self.water_level+0.2)
        new_vegetation = proc_smooth(new_vegetation, self.sigma/2)
        new_vegetation = proc_filter(new_vegetation, self.heightmap, self.water_level)
        self.vegetation += new_vegetation
        self.vegetation[np.where(self.vegetation > 1.0)] = 1.0

        # regrow consumed vegetation
        self.sustenance[np.where((self.sustenance > 0.1) & (self.sustenance < self.vegetation))] += self.s_rate

class Weather:

    def __init__(self, size, storm_chance, storm_size, storm_int, storm_decay, storm_var=(0.75, 1.25), storm_speed=1.0, sigma=2):
        """A weather object, which controls the appearance and movement
           of storms in the simulation.

           Keyword arguments:
           size -- the size of the map.
           storm_chance -- the percent chance a storm will form.
           storm_size -- the average size of a newly formed storm.
           storm_int -- the intensity (damage per second) of a storm.
           storm_decay -- the percent speed at which a storm decays.
           storm_var -- the min/max variance of a storm's size.
           storm_speed -- the average speed of a newly formed storm.
           sigma -- the smoothing factor.
           """

        self.size = size
        self.storm_chance = storm_chance
        self.storm_size = storm_size
        self.storm_int = storm_int
        self.storm_decay = storm_decay
        self.storm_var = storm_var
        self.storm_speed = storm_speed
        self.sigma = sigma
        self.weathermap = np.zeros((size, size))
        self.storms = [] # storms are stored in the format [location, velocity, strength]

    def update(self):
        """Master update function for the weather map."""
        self.weathermap = np.zeros((self.size, self.size))

        # update storms & add them to map
        if(len(self.storms) > 0):
            new_storms = []
            for i in range(len(self.storms)):
                self.storms[i][0][0] += self.storms[i][1][0]
                self.storms[i][0][1] += self.storms[i][1][1]

                # wrap storms around map
                if self.storms[i][0][0] >= self.size:
                    self.storms[i][0][0] = 0
                if self.storms[i][0][1] >= self.size:
                    self.storms[i][0][1] = 0

                if self.storms[i][0][0] < 0:
                    self.storms[i][0][0] = self.size - 1
                if self.storms[i][0][1] < 0:
                    self.storms[i][0][1] = self.size - 1

                self.storms[i][2] *= (1 - self.storm_decay)
                if self.storms[i][2] > 0.1:
                    new_storms.append(self.storms[i])
                self.weathermap[int(self.storms[i][0][0]), int(self.storms[i][0][1])] = self.storms[i][2]


            self.storms = new_storms
            self.weathermap = proc_smooth(self.weathermap, self.sigma)
            self.weathermap = util.mask(self.weathermap, np.average(self.weathermap)*1.2)


        if random.random() < self.storm_chance:
            # generate random position vector for storm
            coords = [random.randint(0, self.size-1), random.randint(0, self.size-1)]
            # generate random movement vector for storm
            direction = [random.uniform(-self.storm_speed, self.storm_speed), random.uniform(-self.storm_speed, self.storm_speed)]
            self.storms.append([coords, direction, 1.0])

    #@jit
    def get_closest_storm(self, coords):
        best_dist = self.size**2
        best = None
        for storm in self.storms:
            dist = util.eudist(storm[0], coords)
            if dist < best_dist:
                best_dist = dist
                best = storm
        return best, best_dist
