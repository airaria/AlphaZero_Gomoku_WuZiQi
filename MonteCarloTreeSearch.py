import numpy as np
import copy

def softmax(x):
    probs = np.exp(x-np.max(x,axis=-1,keepdims=True))
    return probs/np.sum(probs,axis=-1,keepdims=True)

class TreeNode(object):
    def __init__(self,parent,prior_p):
        self.parent = parent
        self.prior_p = prior_p
        self.N = 0
        self.W = 0
        self.Q = 0
        self.U = 0
        self.is_done = False
        self.children = {}  # {action: child, action: child ....}

    def QU(self,c_puct):
        self.U = c_puct*self.prior_p*np.sqrt(self.parent.N)/(1+self.N)
        return self.Q + self.U

    def select(self,c_puct):
        return max(self.children.items(), key=lambda act_child: act_child[1].QU(c_puct))

    def expand(self,action_priors):  # ((action,p),(action,p)...)
        for action,p in action_priors:
            if action not in self.children:
                self.children[action] = TreeNode(self,p)

    def backup(self,v):
        self.N += 1
        self.W += v
        self.Q = self.W/self.N
        if self.parent:
            self.parent.backup(-v)


class MCTS(object):
    def __init__(self, value_fn, c_puct=5, n_search=10000):
        self.root = TreeNode(None,1.)
        self.value_fn = value_fn
        self.c_puct = c_puct
        self.n_search = n_search


    def search(self,game):
        assert self.root.is_done == False
        node = self.root
        action = None
        while len(node.children)>0:
            action,node = node.select(self.c_puct)
            game.fast_step(action)

        # 若当node是黑棋落完子后的局面，那么leaf_value是以白棋的视角衡量的胜率
        action_probs, leaf_value = self.value_fn(game)

        if not action is None:
            done,reward = game.check(action)
            node.is_done = done
            if done:
                leaf_value = reward
        if not node.is_done:
            node.expand(action_priors=action_probs)

        # Update value and visit count of nodes in this traversal.
        node.backup(-leaf_value)

    def update_with_move(self, last_move):
        #if last_move in self.root.children:
        self.root = self.root.children[last_move]
        self.root.parent = None
        #else:
        #    self.root = TreeNode(None, 1.0)

    def get_move_probs(self, game, temperature=1e-3):
        """Runs all playouts sequentially and returns the available actions and their corresponding probabilities
        Arguments:
        state -- the current state, including both game state and the current player.
        temp -- temperature parameter in (0, 1] that controls the level of exploration
        Returns:
        the available actions and the corresponding probabilities
        """
        for n in range(self.n_search):
            env_copy = copy.deepcopy(game)
            self.search(env_copy)

        act_n = [(act, node.N) for act, node in self.root.children.items()]
        acts, visits = zip(*act_n)
        probs = softmax(1.0/temperature * np.log(np.array(visits) + 1e-10))

        return acts, probs


class MCTSPlayer(object):
    def __init__(self,value_fn,c_puct,n_search,is_selfplay,temperature):
        self.mcts = MCTS(value_fn,c_puct,n_search)
        self.is_selfplay = is_selfplay
        self.temperature = temperature

        self.move_count = 0
        self.dir_start = 20

    def reset(self):
        self.move_count = 0
        self.mcts.root = TreeNode (None, 1.0)

    def get_action(self,game,opposite_move=None):
        if opposite_move is not None: # opponent is Human
            self.mcts.update_with_move(opposite_move)

        assert game.n_legal_moves > 0
        acts, probs = self.mcts.get_move_probs(game,temperature=self.temperature)

        self.move_count += 1

        if self.is_selfplay:
            if self.move_count > self.dir_start:
                dirichlet_noise = np.random.dirichlet(0.1 * np.ones(len(probs)))
                move_to_take_i = np.random.choice(len(probs),p=(1-0.25)*probs + 0.25*dirichlet_noise)
            else:
                move_to_take_i = np.random.choice(len(probs),p=probs)
        else:
            move_to_take_i = np.argmax(probs)

        move_to_take = acts[move_to_take_i]
        self.mcts.update_with_move(move_to_take)

        if self.is_selfplay:
            arr_probs = np.zeros(game.height*game.width)
            arr_probs[[a[1] * game.width + a[2] for a in acts]]= probs
            return move_to_take,arr_probs
        else:
            return move_to_take