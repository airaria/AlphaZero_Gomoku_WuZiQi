import numpy as np
from ctypes import *
import numpy.ctypeslib as npct

BLACK = 1
WHITE = 1

lib = npct.load_library("libarb.so","./")
check = lib.check
check.restype = c_int
check.argtypes = [npct.ndpointer(dtype=np.int32,ndim=2),c_int,c_int,c_int]


class Env(object):
    POS_COORD = []

    def __init__(self,size):
        self.size = size
        self.width = size[1]
        self.height = size[0]
        self.board = np.zeros(size,dtype=np.int32)
        self.legal_positions = np.ones(size,dtype=np.bool)
        self.n_legal_moves = self.width * self.height

        self.cur_player = BLACK
        self.last_action = None

        Env.POS_COORD = np.array([(i,j) for i in range(self.height) for j in range(self.width)])

    def reset(self):
        self.board = np.zeros(self.size)
        self.legal_positions = np.ones(self.size,dtype=np.bool)
        self.n_legal_moves = self.width * self.height

        self.cur_player = BLACK
        self.last_action = None

    def fast_step(self,action):   #DOET NOT DO VALIDATION, CALL IT WITH CAUTION!!!
        color, row, col = action
        if self.legal_positions[row,col]==False:
            raise ValueError("Invalid move",row,col)

        self.n_legal_moves -= 1
        self.legal_positions[row,col] = False

        self.cur_player = -self.cur_player
        self.last_action = action
        self.board[row,col] = color

    def step(self,action):
        '''
        :param action: (black(1)/white(-1),x,y)
        :return: board :(width ,height), reward = [-1,0,1], done = True/False
        '''
        self.fast_step(action)
        done,reward = self.check(action)
        return self.board,reward,done

    def check(self,action):
        color, row, col = action
        done = bool(check(self.board,col+row*self.width,self.width,self.height))
        if done:
            reward = color
        else:
            reward = 0
        if self.n_legal_moves == 0:
            done = True
        return done,reward


def make(size):
    return Env(size)

def build_features(game): #cur_player = BLACK(1)/WHITE(-1)

    features = np.zeros((4, game.width, game.height), dtype=np.float32)
    features[0] = game.board == game.cur_player
    features[1] = game.board == -game.cur_player
    if not game.last_action is None:
        lst_player, row, col = game.last_action
        assert lst_player == -game.cur_player
        features[2][row,col] = 1
    features[3,:] = (game.cur_player + 1) / 2
    return features