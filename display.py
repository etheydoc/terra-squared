# SeraphimNashville:Science:AI:ProjectTerra^2//fractal AI + cellular automata
# Class that handles the display
# (c) Seraphim 2018

# Created by Ethan Block
# April 25, 2018

import pygame
from PIL import Image
import numpy as np

def gray(im):
    im = 255 * (im / im.max())
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
    return ret

def terrain_to_surf(tr, scale):
    tr = 255 * tr / tr.max()
    #tr = gray(tr)
    surf = pygame.surfarray.make_surface(tr)
    surf = pygame.transform.scale(surf, (scale, scale))
    return surf

class TSQDisplay:

    def __init__(self, world, res=(1280, 720), cmap=[[0.5, [0.1,0.3,0.4]], [0.6, [0.5,0.5,0.2]], [0.8, [0.3,0.4,0.1]], [0.9,[0.3,0.3,0.25]], [1.1,[0.5, 0.5,0.55]]]):
        pygame.init()
        pygame.display.set_caption("Terra^2")
        self.screen = pygame.display.set_mode(res)
        self.world = world
        self.res = res
        self.cmap = cmap
        # build colored terrain map
        self.terrain_img = np.zeros((self.world.terrain.shape[0], self.world.terrain.shape[1], 3))
        for x in range(self.world.terrain.shape[0]):
            for y in range(self.world.terrain.shape[1]):
                for i in range(len(cmap)):
                    if self.world.terrain[x,y] < cmap[i][0]:
                        self.terrain_img[x,y] = cmap[i][1]
                        break

    def render(self, terrans):
        sf = terrain_to_surf(self.terrain_img, self.res[1])
        self.screen.blit(sf, (0,0))
        pygame.display.flip()

    def update(self, terrans):
        self.render(terrans)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True
