# This class defines the user interface for Terra^2.
# Ethan Block, 10-3-2018

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

import util

def handle_close(evt):
    exit()

class TerraSquaredUI:

    def __init__(self, terrain, weather):

        fig = plt.figure()
        fig.canvas.mpl_connect('close_event', handle_close)

        plt.xlabel("$x$")
        plt.ylabel("$y$")

        bounds = np.array([terrain.v_bounds[0], terrain.v_bounds[1]*0.15, terrain.v_bounds[1]*0.7, terrain.v_bounds[1]*1.4])
        self.norm = colors.BoundaryNorm(boundaries=bounds, ncolors=4)

        self.fig = fig
        self.terrain = terrain
        self.weather = weather

    def update(self, pops=None):
        """Master update function for the UI - redraw the visuals each update.
           """
        plt.clf()
        bg = np.max(self.terrain.heightmap) - self.terrain.heightmap
        plt.imshow(bg, cmap='Blues', interpolation='nearest')

        vmap = util.mask(self.terrain.vegetation, 1e-8)

        plt.imshow(self.norm(vmap), cmap='Greens', alpha=0.8)
        plt.imshow(util.mask(self.terrain.climates, 0.1), cmap='Oranges', interpolation='none', alpha=0.3)

        # render Terran populations
        if pops is not None:
            for i in range(len(pops)):
                if len(pops[i].terrans) > 0:
                    pmap = pops[i].get_positions()[0]
                    pmap = util.mask(pmap, 0.1)
                    plt.imshow(pmap, cmap='Spectral', interpolation='none')

        # render storms
        if np.max(self.weather.weathermap) > 0:
            plt.imshow(util.mask(self.weather.weathermap, 0.1), cmap='Purples', interpolation='none', alpha=0.6)

        plt.draw()
        plt.pause(0.0001)
