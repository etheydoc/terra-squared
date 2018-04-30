# SeraphimNashville:Science:AI:ProjectTerra^2//fractal AI + cellular automata
# Common utilities class
# (c) Seraphim 2018

# Created by Ethan Block
# April 25, 2018

def normalize(x, dmin, dmax, bounds=(0, 1)):
    return bounds[0] + (x - dmin) * (bounds[1] - bounds[0]) / (dmax - dmin)

def denormalize(x, dmin, dmax, bounds=(0, 1)):
    return ((x - bounds[0]) * (dmax-dmin)/(bounds[1]-bounds[0]))+dmin

# Abstract struct class
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
