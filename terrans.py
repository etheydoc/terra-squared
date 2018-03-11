# Definition of the Terrans, the animal inhabitants of Terra^2.
# Ethan Block, 9-3-2018

import numpy as np
import random
import util

from numba import jit

class TerranPop:

    # This class defines an entire population of Terrans (3D array), since defining each one as its own object is too computationally costly.
    def __init__(self, size, pop_size, tmap, dist=1, decay=0.05, sex_th=0.2, decay_health=0.1, decay_social=0.005, temprange=(0.25, 0.75),
                 d_harm=0.2, d_spread=2, w_r=1):
        # size = size of landscape array
        # pop_size = max population size
        # tmap = terrainmap object
        # dist = randomized distance from centroid
        # decay = terran energy decay rate
        # sex_th = energy threshold for pops to reproduce
        # decay_health = health decay rate for weak terrans
        # decay_social = terran social decay rate
        # temprange = survivable temperature range
        # d_harm = damage done to diseased terran per update
        # d_spread = distance a disease can spread
        # w_r = terran weather resistance

        ''' Terrans are stored in this structure:
            [ x coordinate, y coordinate, health value, energy value, social value, resources, rogue, infected, livable temp range ]'''

        pop = [self.newpop(temprange) for x in range(pop_size)]

        # generate above-water coordinates for terrans
        centroid = (0,0)
        while tmap.tmap[centroid] <= 0.5:
            centroid = (random.randint(0, size-1), random.randint(0, size-1))

        for i in range(pop_size):
            pop[i][0] = centroid[0] + random.randint(-dist, dist)

            pop[i][1] = centroid[1] + random.randint(-dist, dist)

        self.size = size
        self.pop = pop
        self.tmap = tmap
        self.decay = decay
        self.sex_th = sex_th
        self.decay_health = decay_health
        self.decay_social = decay_social
        self.temprange = temprange
        self.d_harm = d_harm
        self.d_spread = d_spread
        self.w_r = w_r

    def newpop(self, temprange):
        return [0, 0, 1.0, 1.0, 1.0, 0.0, False, False, temprange]

    @jit
    def get_terrans_in_range(self, coords, radius):
        terran_ids = []
        for terran in self.pop:
            if util.eudist(coords, (terran[0], terran[1])) < radius:
                terran_ids.append(self.pop.index(terran))
        return terran_ids

    @jit
    def get_terran_at_coords(self, coords):
        for terran in self.pop:
            if terran[0] == coords[0] and terran[1] == coords[1]:
                return terran
        return None

    def kill_terran(self, terran):
        for t in self.pop:
            if t == terran:
                self.pop.pop(t)
                break

    @jit
    def get_positions(self):
        pmap = np.zeros((self.size, self.size))
        coords = []
        for i in range(len(self.pop)):
            pmap[int(self.pop[i][0]), int(self.pop[i][1])] = 1.0
            coords.append([int(self.pop[i][0]), int(self.pop[i][1])])
        return pmap, coords

    @jit
    def find_closest_terran(self, coords):
        best = 99999
        best_index = -1

        for i in range(len(self.pop)):
            dist = util.eudist(coords, (self.pop[i][0], self.pop[i][1]))
            if dist < best:
                best = dist
                best_index = i

        return self.pop[best_index]

    @jit
    def find_furthest_terran(self, coords):
        best = 0
        best_index = -1

        for i in range(len(self.pop)):
            dist = util.eudist(coords, (self.pop[i][0], self.pop[i][1]))
            if dist > best:
                best = dist
                best_index = i

        return self.pop[best_index]

    @jit
    def path(self, dest, area=None):
        best = 99999
        best_index = -1

        if area is None:
            area = [[terran[0]-1, terran[1]-1], [terran[0], terran[1]-1], [terran[0]+1, terran[1]-1],
                      [terran[0]-1, terran[1]], [terran[0]+1, terran[1]], [terran[0]-1, terran[1]+1],
                      [terran[0], terran[1]+1], [terran[0]+1, terran[1]+1]]

            # wrap area around map
            for i in range(len(area)):
                for j in range(2):
                    if area[i][j] < 0:
                        area[i][j] = self.size - area[i][j]
                    if area[i][j] >= self.size:
                        area[i][j] = area[i][j] - self.size

        for i in range(len(area)):
            dist = util.eudist(area[i], dest)
            if dist < best:
                best = dist
                best_index = i

        return area[best_index]

    @jit
    def path_away(self, ndest, area=None):
        best = 0
        best_index = -1

        if area is None:
            area = [[terran[0]-1, terran[1]-1], [terran[0], terran[1]-1], [terran[0]+1, terran[1]-1],
                      [terran[0]-1, terran[1]], [terran[0]+1, terran[1]], [terran[0]-1, terran[1]+1],
                      [terran[0], terran[1]+1], [terran[0]+1, terran[1]+1]]

            # wrap area around map
            for i in range(len(area)):
                for j in range(2):
                    if area[i][j] < 0:
                        area[i][j] = self.size - area[i][j]
                    if area[i][j] >= self.size:
                        area[i][j] = area[i][j] - self.size

        for i in range(len(area)):
            dist = util.eudist(area[i], ndest)
            if dist > best:
                best = dist
                best_index = i

        return area[best_index]

    @jit
    def get_centroid(self):
        coords = [0,0]
        for terran in self.pop:
            coords[0] += terran[0]
            coords[1] += terran[1]
        coords[0] /= len(self.pop)
        coords[1] /= len(self.pop)
        return coords

    @jit
    def get_diseased(self):
        num = 0
        for terran in self.pop:
            if terran[7] == True:
                num += 1
        return num

    def update(self):
        #print(len(self.pop))
        to_remove = []
        for terran in self.pop:
            tc = (int(terran[0]), int(terran[1]))
            # calculate terran coords (line-of-sight)
            coords = [[terran[0]-1, terran[1]-1], [terran[0], terran[1]-1], [terran[0]+1, terran[1]-1],
                      [terran[0]-1, terran[1]], [terran[0]+1, terran[1]], [terran[0]-1, terran[1]+1],
                      [terran[0], terran[1]+1], [terran[0]+1, terran[1]+1]]

            # wrap coords around map
            for i in range(len(coords)):
                for j in range(2):
                    #print(coords[i][j])
                    if coords[i][j] < 0:
                        coords[i][j] = self.size - coords[i][j]
                    if coords[i][j] >= self.size:
                        coords[i][j] = coords[i][j] - self.size

            # get closest terran
            closest = self.find_closest_terran(tc)
            cdist = util.eudist((terran[0], terran[1]), (closest[0], closest[1]))

            # reproduction logic (woohoo)
            if terran[3] > self.sex_th:
                if closest[3] > self.sex_th:
                    # subtract energies from terrans
                    terran[3] -= self.sex_th
                    closest[3] -= self.sex_th
                    # create new terran
                    child = self.newpop(self.temprange)
                    child[0] = terran[0]
                    child[1] = closest[1]
                    self.pop.append(child)

            # if on resources and don't have enough, gather them
            if self.tmap.rmap[tc] > 0.1 and terran[5] < 1:
                self.tmap.rmap[tc] -= 0.1
                terran[5] += 0.1

            # social logic
            if terran[4] < 0.5 and cdist > 2:
                terran[2] -= self.decay
                closest = self.find_closest_terran(tc)
                new_coords = self.path((closest[0], closest[1]), area=coords)
                terran[0] = new_coords[0]
                terran[1] = new_coords[1]

            if cdist <= 2:
                terran[4] += self.decay_social
            else:
                # decrement isolated terrans' social level
                terran[4] -= self.decay_social

            # contagion logic
            if terran[7] == True:
                if cdist <= self.d_spread:
                    closest[7] = True

            if terran[4] > 0.5:
                # move away from closest terran and go explore
                new_coords = self.path_away((closest[0], closest[1]), area=coords)

            if self.tmap.smap[tc[0], tc[1]] <= 0.1:
                # decrement all terrans' energy
                terran[3] -= self.decay

            # look for terrans with low energy
            if terran[2] < 0.5:
                # if on vegetation, feed
                if self.tmap.smap[tc[0], tc[1]] > 0.1:
                    terran[2] += self.decay
                    self.tmap.smap[tc[0], tc[1]] -= 0.1
                # otherwise, seek vegetation
                else:
                    # check vegetation gradient
                    best = -1
                    coords_best = coords[0]
                    for i in range(len(coords)):
                        grad = self.tmap.smap[int(coords[i][0]), int(coords[i][1])] - self.tmap.smap[tc[0], tc[1]]
                        if grad > best:
                            best = grad
                            coords_best = (coords[i][0], coords[i][1])

                    # update terran position
                    if best > 0:
                        terran[0] = coords_best[0]
                        terran[1] = coords_best[1]

                    # construction logic



            # decrement weak terrans' health
            if terran[3] <= 0:
                #print('weak damage')
                terran[2] -= self.decay_health

            # heal strong terrans
            elif terran[3] > 0.5 and terran[2] < 1:
                terran[2] += self.decay_health

            # remove dead terrans
            if terran[2] <= 0:
                to_remove.append(self.pop.index(terran))

            # damage drowning terrans
            if self.tmap.tmap[terran[0], terran[1]] <= 0.3:
                #print('drowning damage')
                terran[2] -= 0.1

            # damage terrans in bad climates
            if (self.tmap.cmap[terran[0], terran[1]] < terran[8][0] or
                self.tmap.cmap[terran[0], terran[1]] > terran[8][1]):
                #print(self.tmap.cmap[terran[0], terran[1]])
                #print('climate damage')
                terran[2] -= 0.1

            # damage terrans in storms
            if (self.tmap.wmap[terran[0], terran[1]] > self.w_r):
                # print(self.tmap.wmap[terran[0], terran[1]])
                #print('storm damage')
                terran[2] -= self.tmap.wmap[terran[0], terran[1]]

            # damage diseased terrans
            if (terran[7] == True):
                #print('disease damage')
                terran[2] -= self.d_harm

        # remove terrans
        newpop = []
        for i in range(len(self.pop)):
            if i not in to_remove:
                newpop.append(self.pop[i])
        self.pop = newpop
