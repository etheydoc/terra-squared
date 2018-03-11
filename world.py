# Definition of terrain altitude & vegetation map, and procedural generation algo
# Ethan Block, 9-3-2018

import numpy as np
import scipy as sp
import scipy.ndimage

import random
import util

from numba import jit

import matplotlib.pyplot as pl

class TerrainMap:

    def __init__(self, size, density, v_t=0.2, v_m=1, sigma=3, v_rate=0.01, r_rate=0.005, c_int=0.8, c_sint=0.05, vbounds=(0.2, 0.6),
                 w_int=1, w_decay=0.1, w_sc=0.1, w_sbounds=(0.4, 1.2)):
        # size = terrain map size
        # density = % of map that is land, essentially
        # v_t = vegetation thickness
        # sigma = smoothing factor
        # v_rate = vegetation regrowth rate
        # r_rate = resource regrowth rate
        # c_int = climate intensity %
        # c_sint = climate spawn intensity %
        # vbounds = min/max temperature for vegetation to grow in
        # w_int = weather intensity
        # w_decay = weather storm decay rate
        # w_sc = weather storm spawn chance
        # w_sbounds = min/max weather storm intensity %

        points = int(density*(size**2))

        tmap = np.zeros((size, size))
        tpoints = np.random.choice(range(0, size * size), size=points)
        for i in tpoints:
            tmap[int(i / size), int(i % size)] = 1.0

        tmap = sp.ndimage.filters.gaussian_filter(tmap, [sigma, sigma], mode='constant')
        tmap = util.normalize(tmap, np.min(tmap), np.max(tmap))

        vmap = np.zeros((size, size))
        vpoints = np.random.choice(range(0, size * size), size=int(points*v_t))
        for i in vpoints:
            vmap[int(i / size), int(i % size)] = v_m
        vmap = sp.ndimage.filters.gaussian_filter(vmap, [sigma, sigma], mode='constant')

        rmap = np.zeros((size, size))
        rpoints = np.random.choice(range(0, size * size), size=int(points*v_t))
        for i in vpoints:
            rmap[int(i / size), int(i % size)] = 1.0
        rmap = sp.ndimage.filters.gaussian_filter(rmap, [sigma/2, sigma/2], mode='constant')

        # generate random climates and overlay the north & south pole on top & bottom
        cmap = np.zeros((size, size))
        cpoints = np.random.choice(range(0, size * size), size=int(points*c_sint))
        for i in cpoints:
            cmap[int(i / size), int(i % size)] = random.random()
        cmap[0:int(size/8)] = [0.0] * size
        cmap[-int(size/8):-1] = [0.0] * size
        cmap = sp.ndimage.filters.gaussian_filter(cmap, [sigma*2, sigma*2], mode='constant')
        cmap = util.normalize(cmap, np.min(cmap), np.max(cmap))
        # print(np.max(cmap))

        # weather map - tracks storms and stuff
        self.wmap = np.zeros((size, size))

        # masking - remove vegetation below water level and in cold/hot areas
        for x in range(size):
            for y in range(size):
                if tmap[x][y] <= 0.5:
                    vmap[x][y] = 0
                else:
                    #cval = cmap[x][y] - 0.5
                    cval = cmap[x][y]
                    #print(cval)
                    if cval > vbounds[0] and cval < vbounds[1]:
                        vmap[x][y] *= cval

        vmap = util.normalize(vmap, np.min(vmap), np.max(vmap))
        rmap = util.normalize(rmap, np.min(rmap), np.max(rmap))

        self.size = size
        self.sigma = sigma
        self.tmap = tmap
        self.vmap = vmap
        self.smap = np.copy(vmap) # sustenance map
        self.rmap = rmap
        self.cmap = cmap
        self.v_rate = v_rate
        self.r_rate = r_rate
        self.w_int = w_int
        self.w_decay = w_decay
        self.w_sc = w_sc
        self.w_sbounds = w_sbounds

    def update_weather(self):
        wmap = self.wmap.T
        cached = wmap[-1]
        wmap = util.shift(wmap, 1, fill_value=cached).T

        wmap -= (wmap * self.w_decay)

        # spawn storm
        if random.random() < self.w_sc:
            nwmap = np.zeros((self.size, self.size))
            wpoints = np.random.choice(range(0, self.size * self.size), size=1)
            for i in wpoints:
                nwmap[int(i / self.size), int(i % self.size)] = 1.0
            nwmap = sp.ndimage.filters.gaussian_filter(nwmap, [self.sigma, self.sigma], mode='constant')
            nwmap *= self.w_int * random.uniform(self.w_sbounds[0], self.w_sbounds[1])
            nwmap = util.normalize(nwmap, 0, np.max(nwmap))
            wmap += nwmap

        #if np.max(wmap) > 0:
            #wmap = util.normalize(wmap, np.min(wmap), np.max(wmap))


        #print(np.min(wmap), np.max(wmap))# debug
        self.wmap = wmap

    @jit
    def grow_vegetation(self):
        for x in range(self.size):
            for y in range(self.size):
                if self.rmap[x,y] > 0:
                    if self.smap[x,y] < self.vmap[x,y]:
                        self.smap[x,y] = min(self.smap[x,y]+self.v_rate, self.vmap[x,y])
                    # grow resources
                    if self.rmap[x,y] < 1:
                        self.rmap[x,y] = min(self.rmap[x,y]+self.r_rate, 1)
