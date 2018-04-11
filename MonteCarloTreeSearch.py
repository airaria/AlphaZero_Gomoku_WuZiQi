import numpy as np
import copy
from config import *

def softmax(x):
    probs = np.exp(x-np.max(x,axis=-1,keepdims=True))
    return probs/np.sum(probs,axis=-1,keepdims=True)

class TreeNode(object):
    def __init__(self,parent,prior_p):
        self.parent = parent
        self.prior_p = prior_p
        self.N = 0
        self.W = 0
        self.is_done = False
        self.children = {}  # {action: child, action: child ....}
        self.is_expanded = False

    def QU(self,c_puct):
        U = c_puct*self.prior_p*np.sqrt(self.parent.N)/(1+self.N)
        if self.N == 0: # Q = 0
            return U
        return self.W/self.N + U

    def select(self,c_puct):
        return max(self.children.items(), key=lambda act_child: act_child[1].QU(c_puct))

    def expand(self,action_priors):  # ((action,p),(action,p)...)
        if self.is_expanded:
            return #self.revert_visits()
        else:
            self.is_expanded = True
            for action,p in action_priors:
                if action not in self.children:
                    self.children[action] = TreeNode(self,p)

    def expand_and_backup(self,action_priors,value):
        if self.is_expanded:
            return #self.revert_visits()
        else:
            self.is_expanded = True
            for action,p in action_priors:
                if action not in self.children:
                    self.children[action] = TreeNode(self,p)
            self.backup(value)

    def backup(self,v):
        self.N += 1
        self.W += v
        if self.parent:
            self.parent.backup(-v)

    '''
    def revert_visits(self):
        self.N -= 1
        if self.parent:
            self.parent.revert_visits()
    '''

    def add_virtual_loss(self):
        self.N += N_VIRTUAL_LOSS
        self.W -= N_VIRTUAL_LOSS
        if self.parent:
            self.parent.add_virtual_loss()

    def revert_virtual_loss(self):
        self.N -= N_VIRTUAL_LOSS
        self.W += N_VIRTUAL_LOSS
        if self.parent:
            self.parent.revert_virtual_loss()

class MCTS(object):
    def __init__(self, value_fn, c_puct=2, n_search=1000):
        self.root = TreeNode(None,1.)
        self.value_fn = value_fn
        self.c_puct = c_puct
        self.n_search = n_search

    def search_many(self,origin_game,many=N_EVALUATE):
        #assert self.root.is_done == False
        nodes = []
        games = []
        for _ in range(many):
            action = None
            game = copy.deepcopy(origin_game)

            node = self.root
            while len(node.children)>0:
                action,node = node.select(self.c_puct)
                game.fast_step(action)

            if not action is None:
                node.is_done, reward = game.check(action)
            if node.is_done:
                leaf_value = -1  # reward
                node.backup(-leaf_value)
            else:
                node.add_virtual_loss()
                games.append(game)
                nodes.append(node)

        if len(games)>0:
            # 若node是黑棋落完子后的局面，那么leaf_value是以白棋的视角衡量的胜率
            action_probs, leaf_values = self.value_fn(games,many=True)
            for node,action_prob,leaf_value in zip(nodes,action_probs,leaf_values):
                #assert not node.is_done
                node.revert_virtual_loss()
                node.expand_and_backup(action_priors=action_prob,value=-leaf_value)
                #node.expand(action_priors=action_prob)
                #if not node.is_expanded:
                #    node.backup(-leaf_value)

    def search(self,origin_game):
        game = copy.deepcopy(origin_game)
        assert self.root.is_done == False
        node = self.root
        action = None
        while len(node.children)>0:
            action,node = node.select(self.c_puct)
            game.fast_step(action)
        # 若当node是黑棋落完子后的局面，那么leaf_value是以白棋的视角衡量的胜率
        action_probs, leaf_value = self.value_fn(game)

        if not action is None:
            node.is_done,reward = game.check(action)
        if node.is_done:
                leaf_value = -1 #reward
        else:
            node.expand(action_priors=action_probs)

        # Update value and visit count of nodes in this traversal.
        node.backup(-leaf_value)

    def update_with_move(self, last_move):
        #if last_move in self.root.children:
        self.root = self.root.children[last_move]
        self.root.parent = None
        #else:
        #    self.root = TreeNode(None, 1.0)

    def get_move_probs(self, game, temperature):

        for n in range(self.n_search):
            self.search_many(game)

        act_n = [(act, node.N) for act, node in self.root.children.items()]
        acts, visits = zip(*act_n)
        probs = softmax(1.0/temperature * np.log(np.array(visits) + 1e-10))

        return acts, probs


class MCTSPlayer(object):
    def __init__(self,controller,c_puct,n_search,return_probs,temperature,noise=True):
        self.controller = controller
        self.mcts = MCTS(controller.value_fn,c_puct,n_search)
        self.return_probs = return_probs
        self.temperature = temperature
        self.noise = noise
        self.game = None
        self.acts = None
        self.probs = None

        self.move_count = 0
        self.dir_start = 20

    def reset(self):
        self.move_count = 0
        self.mcts.root = TreeNode (None, 1.0)

    def observe(self,game,opposite_move=None):
        self.game = game
        if not opposite_move is None:
            if opposite_move not in self.mcts.root.children:
                print ("Not in AI's mc tree...",opposite_move)
            else:
                self.mcts.update_with_move(opposite_move)

    def think(self):
        #assert self.game.n_legal_moves > 0
        self.acts, self.probs = self.mcts.get_move_probs(self.game, temperature=self.temperature)
        if self.return_probs:
            arr_probs = np.zeros(self.game.height * self.game.width)
            arr_probs[[a[1] * self.game.width + a[2] for a in self.acts]] = self.probs
            return arr_probs

    def take_action(self):
        self.move_count += 1
        if self.noise:
            if self.move_count > self.dir_start:
                dirichlet_noise = np.random.dirichlet(0.1 * np.ones(len(self.probs)))
                move_to_take_i = np.random.choice(len(self.probs),p=(1-0.25)*self.probs + 0.25*dirichlet_noise)
            else:
                move_to_take_i = np.random.choice(len(self.probs),p=self.probs)
        else:
            move_to_take_i = np.argmax(self.probs)

        move_to_take = self.acts[move_to_take_i]
        self.mcts.update_with_move(move_to_take)

        return move_to_take
