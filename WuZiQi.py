import numpy as np
from ctypes import *
import numpy.ctypeslib as npct
from config import BOARD_SIZE

BLACK = 1
WHITE = -1

lib = npct.load_library("libarb.so","./")
check = lib.check
check.restype = c_int
check.argtypes = [npct.ndpointer(dtype=np.int32,ndim=2),c_int,c_int,c_int]


class Env(object):
    POS_COORD = np.array([(i,j) for i in range(BOARD_SIZE[0]) for j in range(BOARD_SIZE[1])])

    def __init__(self,size):
        self.size = size
        assert self.size == BOARD_SIZE
        self.width = size[1]
        self.height = size[0]
        self.N = self.width * self.height
        self.board = np.zeros(size,dtype=np.int32)
        self.legal_positions = np.ones(size,dtype=np.bool)
        self.n_legal_moves = self.width * self.height

        self.cur_player = BLACK
        self.last_action = None


    def reset(self):
        self.board = np.zeros(self.size,dtype=np.int32)
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
        #print (self.board, col + row * self.width, self.width, self.height)
        res = check(self.board, col + row * self.width, self.width, self.height)
        done = bool(res)
        if done:
            reward = color
        else:
            reward = 0
        if self.n_legal_moves == 0:
            done = True
        return done,reward

    def build_features(self,rot_and_flip=False): #cur_player = BLACK(1)/WHITE(-1)
        features = np.zeros((4, self.width, self.height), dtype=np.float32)
        features[0] = self.board == self.cur_player
        features[1] = self.board == -self.cur_player
        if not self.last_action is None:
            lst_player, row, col = self.last_action
            #assert lst_player == -self.cur_player
            features[2][row,col] = 1
        features[3,:] = (self.cur_player + 1) / 2

        if rot_and_flip:  #random rotation | flipping
            rot_angle = np.random.randint(4)
            flip_flag = np.random.random()
            features = np.rot90(features,rot_angle,axes=(1,2))
            if flip_flag>0.5:
                features = np.flip(features,axis=-1)
            return features.copy(),(rot_angle,flip_flag)
        else:
            return features

    def __repr__(self):
        black = '●'
        white = '○'
        sym_dict = {1:black,0:' ',-1:white}
        sym_board = []
        for i,row in enumerate(self.board):
            sym_board.append(" {:02d} ┃ ".format(i) + ' | '.join([sym_dict[j] for j in row])+' |')
        sym_board = '\n'.join(sym_board)
        return \
        (" {}  ┃ ".format(sym_dict[self.cur_player]) + '| '.join(["{:02d}".format(i) for i in range(self.height)])+'|'+'\n')\
        +'\n'\
        +sym_board
