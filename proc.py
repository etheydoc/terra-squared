# SeraphimNashville:Science:AI:ProjectTerra^2//fractal AI + cellular automata
# This file contains a bunch of methods for procedural generation
# (c) Seraphim 2018

# Created by Ethan Block
# April 25, 2018

import numpy as np
import scipy as sp
import scipy.ndimage
import random
import util

def gen_climates(size, num_climates=3, sigma=3):
    climates = np.zeros((size, size))
    for i in range(num_climates):
        climates[random.randint(0, size-1), random.randint(0, size-1)] = 1.0
    return smooth(climates,sigma=sigma)

def gen_terrain(size, seed=0, sigma=3):
    lin = np.linspace(0,size,endpoint=False)
    x,y = np.meshgrid(lin,lin)
    return smooth(perlin(x,y,seed=seed), sigma=sigma)

def smooth(x, sigma=3):
    y = sp.ndimage.filters.gaussian_filter(x, [sigma, sigma], mode='constant')
    y = util.normalize(y, np.min(y), np.max(y))
    return y

def perlin(x,y,seed=0):
    # permutation table
    np.random.seed(seed)
    p = np.arange(256,dtype=int)
    np.random.shuffle(p)
    p = np.stack([p,p]).flatten()
    # coordinates of the top-left
    xi = x.astype(int)
    yi = y.astype(int)
    # internal coordinates
    xf = x - xi
    yf = y - yi
    # fade factors
    u = fade(xf)
    v = fade(yf)
    # noise components
    n00 = gradient(p[p[xi]+yi],xf,yf)
    n01 = gradient(p[p[xi]+yi+1],xf,yf-1)
    n11 = gradient(p[p[xi+1]+yi+1],xf-1,yf-1)
    n10 = gradient(p[p[xi+1]+yi],xf-1,yf)
    # combine noises
    x1 = lerp(n00,n10,u)
    x2 = lerp(n01,n11,u) # FIX1: I was using n10 instead of n01
    return lerp(x1,x2,v) # FIX2: I also had to reverse x1 and x2 here

def lerp(a,b,x):
    "linear interpolation"
    return a + x * (b-a)

def fade(t):
    "6t^5 - 15t^4 + 10t^3"
    return 6 * t**5 - 15 * t**4 + 10 * t**3

def gradient(h,x,y):
    "grad converts h to the right gradient vector and return the dot product with (x,y)"
    vectors = np.array([[0,1],[0,-1],[1,0],[-1,0]])
    g = vectors[h%4]
    return g[:,:,0] * x + g[:,:,1] * y
