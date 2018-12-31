# SeraphimNashville:Science:AI:ProjectTerra^2//fractal AI + cellular automata
# Main class that ties everything together & runs it
# (c) Seraphim 2018

# Created by Ethan Block
# April 25, 2018

from display import TSQDisplay
from world import TSQWorld
from terrans import TerranPop
import util

class TerraSquared:

    def __init__(self, size=128, num_pops=4, size_pops=4):
        self.world = TSQWorld(size=size)
        self.num_pops = num_pops
        self.display = TSQDisplay(self.world)
        self.running = True
        self.terrans = [TerranPop(size_pops, self.world, pos=self.world.find_spawn(), bounds=(size,size)) for i in range(num_pops)]

    def run(self):
        # debug
        #print(self.terrans[0].terrans[1].x, self.terrans[0].terrans[1].y)

        while self.running:
            for i in range(self.num_pops):
                self.terrans[i].update()
            self.running = self.display.update(self.terrans)

if __name__ == '__main__':
    tsq = TerraSquared(num_pops=9, size_pops=3, size=256)
    tsq.run()
