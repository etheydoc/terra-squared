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

    def __init__(self, size=128, num_pops=4):
        self.world = TSQWorld(size=size)
        self.display = TSQDisplay(self.world)
        self.running = True
        self.terrans = [TerranPop(4) for i in range(num_pops)]

    def run(self):
        while self.running:

            self.running = self.display.update(self.terrans)

if __name__ == '__main__':
    tsq = TerraSquared(size=256)
    tsq.run()
