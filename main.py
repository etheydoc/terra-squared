# Main class of Terra^2
# Ethan Block, 9-3-2018

import world, terrans, util
import matplotlib.pyplot as pl
import numpy as np
import random

class TerraSquared:

    def __init__(self, size=32, density=0.15, pop_size=8, v_t=0.2, v_m=1, sex_th=0.2, decay=0.025, decay_health=0.05, decay_social=0.005, v_rate=0.002,
                 temprange=(0.25, 0.75), vbounds=(0.2, 0.6), w_int=1, d_harm=0.4, d_radius=5, d_of=8, d_spread=2, d_chance=0.001, c_int=0.4, w_sc=0.025):

        # d_radius = radius around centroid that checks for overpopulation
        # d_of = disease overpopulation factor, how many terrans need to be in the radius to start a disease
        # d_chance = chance for a disease to start %

        # terrain map for terrans to live in
        tmap = world.TerrainMap(size, density, v_t, v_m, v_rate=v_rate, vbounds=vbounds, w_int=w_int, c_int=c_int, w_sc=w_sc)

        # terrans - population of the simulation
        pop1 = terrans.TerranPop(size, pop_size, tmap, sex_th=sex_th, decay=decay, decay_health=decay_health, decay_social=decay_social, temprange=temprange,
                                 d_harm=d_harm, d_spread=d_spread)
        #pop2 = terrans.TerranPop(size, pop_size, tmap, sex_th=sex_th, decay=decay, decay_health=decay_health, decay_social=decay_social, temprange=temprange)
        pop2 = terrans.TerranPop(size, pop_size, tmap, sex_th=sex_th*1.25, decay=decay*0.8, decay_health=decay_health*1.1, decay_social=decay_social*1.1,
                                 temprange=(temprange[0]*1.1, temprange[1]*0.9), d_harm=d_harm, d_spread=d_spread)

        pl.xlabel("$x$")
        pl.ylabel("$y$")

        while True:
            if len(pop1.pop) > 0:
                # pop1 - go rogue
                p1c = pop1.get_centroid()
                furthest1 = pop1.find_furthest_terran(p1c)
                if furthest1[5] == False:
                    if furthest1[3] <= 0.1:
                        pop2.pop.append(furthest1)
                        pop1.pop.pop(pop1.pop.index(furthest1))
                        furthest1[5] = True

                pop1.update()

                # spawn diseases in population centers
                tids1 = pop1.get_terrans_in_range(p1c, d_radius)
                if random.random() < d_chance:
                    if len(tids1) >= d_of:
                        print("Disease started in pop1")
                        index = tids1[random.randint(0, len(tids1)-1)]
                        pop1.pop[index][7] = True

            # go rogue if too far away and antisocial
            if len(pop2.pop) > 0:
                p2c = pop2.get_centroid()
                furthest2 = pop2.find_furthest_terran(p2c)
                if furthest2[5] == False:
                    if furthest2[3] <= 0.1:
                        pop1.pop.append(furthest2)
                        pop2.pop.pop(pop2.pop.index(furthest2))
                        furthest2[5] = True

                pop2.update()

                # spawn diseases in population centers
                tids2 = pop2.get_terrans_in_range(p2c, d_radius)
                if random.random() < d_chance:
                    if len(tids2) >= d_of:
                        print("Disease started in pop2")
                        index = tids2[random.randint(0, len(tids2)-1)]
                        pop2.pop[index][7] = True


            pmap, pcoords = pop1.get_positions()
            pmap2, pcoords2 = pop2.get_positions()

            # combat time
            if len(pop1.pop) > 0 and len(pop2.pop) > 0:
                combats = np.where(pcoords == pcoords2)[0]
                if len(combats) > 0:
                    for i in range(len(combats)):
                        c1 = pop1.get_terran_at_coords(combats[i])
                        c2 = pop2.get_terran_at_coords(combats[i])

                        if c1[3] > c2[3]:
                            c2[3] -= c1[3]
                            pop1.kill_terran(c1)
                        elif c1[3] > c2[3]:
                            c1[3] -= c2[3]
                            pop2.kill_terran(c2)
                        else:
                            pop1.kill_terran(c1)
                            pop2.kill_terran(c2)

            #pmap += pmap2
            #pmap[np.where(pmap>1)]=1

            pl.clf()
            bg = np.max(tmap.tmap) - tmap.tmap
            pl.imshow(bg, cmap='Blues', interpolation='nearest')
            # alpha blending
            if np.max(tmap.smap) > 0.2:
                smap = np.ma.masked_where(tmap.smap < 0.2, tmap.smap)
                pl.imshow(smap, cmap='Greens', interpolation='none')

            cmap = np.ma.masked_where(tmap.cmap < 0.05, tmap.cmap)
            pl.imshow(tmap.cmap, cmap='Oranges', interpolation='nearest', alpha=0.5)

            if np.max(tmap.wmap) > 0.01:
                wmap = np.ma.masked_where(tmap.wmap < 0.005, tmap.wmap)
                pl.imshow(wmap, cmap='Purples', interpolation='nearest', alpha=0.25)

            if np.max(pmap) > 0:
                pmap = np.ma.masked_where(pmap < 1, pmap)
                pl.imshow(pmap, cmap='prism', interpolation='none')

            if np.max(pmap2) > 0:
                pmap2 = np.ma.masked_where(pmap2 < 1, pmap2)
                pl.imshow(pmap2, cmap='Spectral', interpolation='none')

            pl.draw()
            pl.pause(0.01)

            tmap.grow_vegetation()
            tmap.update_weather()

if __name__ == '__main__':
    tsq = TerraSquared(size=64, pop_size=3, sex_th=0.2, decay=0.07, density=0.5, v_t=0.3, decay_social=0.001, v_rate=0.001, temprange=(0.15, 0.85), w_int=0.8)
