# Common utilities class for the Terra^2 simulation.
# Ethan Block, 9-3-2018

from numba import jit
import numpy as np

# abstract Struct class
class Struct:
    def __init__ (self, *argv, **argd):
        if len(argd):
            # Update by dictionary
            self.__dict__.update (argd)
        else:
            # Update by position
            attrs = filter (lambda x: x[0:2] != "__", dir(self))
            for n in range(len(argv)):
                setattr(self, attrs[n], argv[n])

def normalize(x, dmin=None, dmax=None, bounds=(0, 1)):
    if dmin is None:
        dmin = np.min(x)
    if dmax is None:
        dmax = np.max(x)

    return bounds[0] + (x - dmin) * (bounds[1] - bounds[0]) / (dmax - dmin)

def denormalize(x, dmin, dmax, bounds=(0, 1)):
    return ((x - bounds[0]) * (dmax-dmin)/(bounds[1]-bounds[0]))+dmin

def mask(to_mask, threshold):
    if(np.max(to_mask) > threshold):
        to_mask = np.ma.masked_where(to_mask < threshold, to_mask)
    return to_mask

@jit
def get_area(coords, size):
    """Returns a 3x3 area around the supplied coordinates.

       Keyword arguments:
       coords -- the coordinates to calculate the area around.
       size -- the size of the terrain map (for wrapping).
       """

    area = [[coords[0]-1, coords[1]-1], [coords[0], coords[1]-1], [coords[0]+1, coords[1]-1],
            [coords[0]-1, coords[1]], [coords[0], coords[1]], [coords[0]+1, coords[1]],
            [coords[0]-1, coords[1]+1], [coords[0], coords[1]+1], [coords[0]+1, coords[1]+1]]

    # wrap area around map
    for i in range(len(area)):
        for j in range(2):
            if area[i][j] < 0:
                area[i][j] = size - area[i][j]
            if area[i][j] >= size:
                area[i][j] = area[i][j] - size

    return area

@jit
def get_gradient(inmap):
    """Returns a 3D map describing the gradient of the input map.

       Keyword arguments:
       inmap -- the map to create a gradient from.
       """

    size = inmap.shape[0]
    gradient = np.zeros((size, size, 9))
    for x in range(size):
        for y in range(size):
            area = get_area((x, y), size)
            for z in range(9):
                gradient[x, y, z] = ((inmap[area[z][0], area[z][1]]-inmap[x,y]))
    return gradient

@jit
def path(dest, area):
    best = 99999
    best_index = -1

    for i in range(len(area)):
        dist = eudist(area[i], dest)
        if dist < best:
            best = dist
            best_index = i

    return area[best_index]

@jit
def path_away(ndest, area):
    best = 0
    best_index = -1

    for i in range(len(area)):
        dist = eudist(area[i], ndest)
        if dist > best:
            best = dist
            best_index = i

    return area[best_index]

@jit
def shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result = arr
    return result

#@jit DEBUG: was causing crashes cause of JIT
def eudist(v1, v2):
    if not len(v1) == len(v2):
        raise Exception("Inequal vector lengths")
    dist = 0
    for i in range(len(v1)):
        dist += (v2[i] - v1[i])**2
    dist = dist**(1/2)
    return dist

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
