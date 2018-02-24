import WuZiQi
import numpy as np
import MonteCarloTreeSearch
import Network

BOARD_SIZE = (7,7)
BLACK = 1
WHITE = -1

if __name__=='__main__':
    game = WuZiQi.Env(BOARD_SIZE)