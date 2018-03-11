# Common utilities class for the Terra^2 simulation.
# Ethan Block, 9-3-2018

from numba import jit
import numpy as np

def normalize(x, dmin, dmax, bounds=(0, 1)):
    return bounds[0] + (x - dmin) * (bounds[1] - bounds[0]) / (dmax - dmin)

def denormalize(x, dmin, dmax, bounds=(0, 1)):
    return ((x - bounds[0]) * (dmax-dmin)/(bounds[1]-bounds[0]))+dmin

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

@jit
def eudist(v1, v2):

    if not len(v1) == len(v2):
        raise Exception("Inequal vector lengths")

    dist = 0
    for i in range(len(v1)):
        dist += (v2[i] - v1[i])**2
    dist = dist**(1/2)
    return dist
