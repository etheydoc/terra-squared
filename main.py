# Main class of Terra^2
# Ethan Block, 10-3-2018

import display, terrans, world
import msvcrt

class TerraSquared:

    def __init__(self, size=32, points=None, delay=100, num_terrans=4, spawn_dist=2, temprange=(0.0, 1.0),
                 storm_chance=0.01, storm_size=1, storm_int=0.2, storm_decay=0.05, storm_var=(0.8, 1.2), storm_speed=2.0):
        """The main class for the Terra^2 simulation.

           Keyword arguments:
           size -- the size of the generated terrain.
           points -- the number of high-altitude points in the terrain.
           delay -- the amount of steps to pass before spawning in Terrans.
           num_terrans -- the initial amount of Terrans to spawn.
           spawn_dist -- the maximum distance from the spawn point a Terran may be placed.
           temprange -- the survivable temperature range for Terrans.
           """

        if points is None:
            points = size*12

        self.delay = delay
        self.num_terrans = num_terrans
        self.spawn_dist = spawn_dist
        self.temprange = temprange

        self.spawned = False
        self.terrain = world.Terrain(size, points=points)
        self.weather = world.Weather(size, storm_chance, storm_size, storm_int, storm_decay, storm_var=storm_var, storm_speed=storm_speed)
        self.ui = display.TerraSquaredUI(self.terrain, self.weather)

    def run(self):
        step = 0
        while True:
            self.terrain.update()
            self.weather.update()

            if not self.spawned:
                if step >= self.delay:
                    self.tpop = terrans.TerranPop(self.terrain, self.weather, self.num_terrans, temprange=self.temprange, spawn_dist=self.spawn_dist, sex_th=0.3)
                    self.spawned = True
                self.ui.update()
            else:
                self.ui.update(pops=[self.tpop])
                self.tpop.update()

            # increment step counter
            step += 1

            # check for terminating keystroke
            if msvcrt.kbhit():
                if ord(msvcrt.getch()) == 27:
                    break

if __name__ == '__main__':
    tsq = TerraSquared(size=64)
    tsq.run()
