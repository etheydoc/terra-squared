# Definition of the Terrans, the animal inhabitants of Terra^2.
# Ethan Block, 10-3-2018

from numba import jit
from util import Struct
import world, util

import numpy as np
import random

class Terran(Struct):
    """The struct that contains variables controlling
       the behavior of a Terran, a single animal inhabitant
       of Terra^2."""
    x = 0
    y = 0
    health = 1.0
    energy = 1.0
    social = 1.0
    rogue = False
    infected = False
    temprange = (0.0, 1.0)
    w_climate = 1.0 # how much the Terran cares about climate
    w_vegetation = 1.0 # how much the Terran cares about vegetation

class TerranPop:

    def __init__(self, terrain, weather, num_terrans, spawn_dist=2, temprange=(0.0, 1.0), decay=0.1, decay_h=0.25, decay_soc=0.005, sex_th=0.4):
        """The class that defines a population of Terrans and
           controls their behavior.

           Keyword arguments:
           terrain -- the world object these Terrans inhabit.
           weather --  the weather controller of the world.
           num_terrans -- the initial number of Terrans in this population.
           spawn_dist -- the maximum distance from the spawn point a Terran may be placed.
           temprange -- the range of survivable temperatures for the initial Terrans in this population.
           decay -- the decay rate of an individual's energy.
           decay_h -- the decay rate of an individual's health when exhausted.
           decay_soc -- the decay rate of a Terran's social need.
           sex_th -- the energy threshold required for two Terrans to reproduce.
           """

        self.temprange = temprange
        self.decay = decay
        self.decay_h = decay_h
        self.decay_soc = decay_soc
        self.sex_th = sex_th

        spawn_point = (int(terrain.size/2), int(terrain.size/2))
        while (terrain.climates[spawn_point] < temprange[0] or
               terrain.climates[spawn_point] > temprange[1] or
               terrain.heightmap[spawn_point] <= terrain.water_level or
               terrain.vegetation[spawn_point] < np.max(terrain.vegetation)/2):

               spawn_point = (random.randint(0, terrain.size-1), random.randint(0, terrain.size-1))

        self.terrain = terrain
        self.weather = weather
        self.gradient_c = util.get_gradient(world.proc_smooth(terrain.climates, 3) - abs((temprange[1] - temprange[0])/2))
        for x in range(self.terrain.size):
            for y in range(self.terrain.size):
                self.gradient_c[x,y] = util.normalize(self.gradient_c[x,y])

        self.terrans = [Terran(x=(spawn_point[0] + random.randint(-spawn_dist, spawn_dist)),
                        y=(spawn_point[1] + random.randint(-spawn_dist, spawn_dist)),
                        temprange=temprange) for x in range(num_terrans)]
        self.terran_coords = np.array(self.get_positions()[1])

    #@jit
    def get_positions(self):
        pmap = np.zeros((self.terrain.size, self.terrain.size))
        coords = []
        for terran in self.terrans:
            pmap[terran.x, terran.y] = 1.0
            coords.append((terran.x, terran.y))
        return pmap, coords

    #@jit
    def get_closest_terran(self, coords):
        best = None
        best_dist = 99999

        for i in range(len(self.terran_coords)):
            dist = util.eudist(coords, self.terran_coords[i])
            if dist < best_dist:
                best = self.terrans[i]
                best_dist = dist

        return best, best_dist

    def manage_terrans(self):
        newpops = []
        for terran in self.terrans:
            if terran.energy < 1.0:
                # consume sustenance
                if self.terrain.sustenance[terran.x, terran.y] > 0:
                    self.terrain.sustenance[terran.x, terran.y] -= self.decay*2
                    terran.energy += self.decay*2

            # damage weak terrans
            if terran.energy <= 0.0:
                terran.health -= self.decay_h

            # damage terrans in storm
            if self.weather.weathermap[terran.x, terran.y] > 0:
                terran.health -= self.weather.weathermap[terran.x, terran.y]

            # remove dead terrans
            if terran.health > 0.0:
                newpops.append(terran)

            # social and reproduction logic (woohoo)
            closest, closest_dist = self.get_closest_terran([terran.x, terran.y])
            if closest_dist <= 2:
                terran.social += self.decay_soc
                if closest.energy > self.sex_th and terran.energy > self.sex_th:
                    self.terrans.append(Terran(x=terran.x, y=closest.y, temprange=self.temprange))
                    closest.energy -= self.sex_th
                    terran.energy -= self.sex_th
            else:
                terran.social -= self.decay_soc

            if terran.social <= 0.0:
                terran.energy -= self.decay_h

            terran.energy -= self.decay

        self.terrans = newpops

    def move_terrans(self):
        self.terran_coords = np.array(self.get_positions()[1])
        # TODO: calculate area gradients for each terran, move accordingly
        for terran in self.terrans:
            txy = np.array([terran.x, terran.y])
            closest = self.get_closest_terran(txy)[0]
            closest_coords = (closest.x ,closest.y)
            area = util.get_area(txy, self.terrain.size)
            grad = np.zeros(9)

            for i in range(len(area)):
                g_sust = self.terrain.sustenance[area[i][0], area[i][1]] - self.terrain.sustenance[txy[0], txy[1]]
                g_cli = np.average(self.gradient_c[area[i][0], area[i][1]]) - self.gradient_c[txy[0], txy[1]][4]
                grad[i] = (g_sust + g_cli) / 2

            dest = area[grad.tolist().index(max(grad))]
            dest = np.asarray(dest)

            if terran.social < 0.2:
                dest_soc = util.path(closest_coords, area)
                dest = (dest + dest_soc) / 2
            elif terran.social > 0.8:
                dest_soc = util.path_away(closest_coords, area)
                dest = (dest + dest_soc) / 2

            if len(self.weather.storms) > 0:
                if self.weather.weathermap[txy[0], txy[1]] > 0:
                    storm = self.weather.get_closest_storm(txy)[0][0]
                    g_storm = util.path_away(storm, area)
                    dest = (dest + g_storm) / 2

            dest = np.ndarray.astype(np.asarray(dest), 'int32')

            # find a spot near destination that isn't occupied or below sea level
            dest_area = util.get_area(dest, self.terrain.size)
            for i in range(len(dest_area)):
                if dest_area[i] not in self.terran_coords:
                    if self.terrain.heightmap[dest_area[i][0], dest_area[i][1]] > self.terrain.water_level:
                        terran.x = dest[0]
                        terran.y = dest[1]
                        break

    def update(self):
        self.move_terrans()
        self.manage_terrans()
