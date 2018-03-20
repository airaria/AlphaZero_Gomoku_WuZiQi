import WuZiQi
import numpy as np
import MonteCarloTreeSearch
import Network
import copy
from config import *
from utils import data_augment,Buffer
import torch.multiprocessing as mp
from queue import Empty as EmptyError
import time

def self_play(game,AIplayer):
    game = copy.deepcopy(game)
    game.reset()
    AIplayer.reset()
    states_list, probs_list, current_player_list = [], [], []
    is_done = False
    reward = 0
    while True:
        if is_done:
            break

        print(game)

        AIplayer.observe(game)
        probs = AIplayer.think()

        print(probs.max(), probs.argmax() // 7, probs.argmax() % 7)

        move_to_take = AIplayer.take_action()
        print (move_to_take)

        states_list.append(game.build_features(rot_and_flip=0))
        probs_list.append(probs)
        current_player_list.append(game.cur_player)

        _, reward, is_done = game.step(move_to_take)
        print ('is_done',is_done)
    print(game)
    win_list = [cp*reward for cp in current_player_list]
    #augmentation
    states_a, probs_a, wins_a = data_augment(states_list,probs_list,win_list)
    #add to buffer
    return states_a, probs_a, wins_a


def human_play(game,AIplayer,BW):
    game = copy.deepcopy(game)
    game.reset()
    AIplayer.reset()
    AIplayer.noise = False
    is_done = False
    last_human_action = None

    while True:
        if is_done:
            break
        print (game)
        if game.cur_player == BW:
            posstr= input("action: x,y").split(',')
            move_to_take = (game.cur_player,int(posstr[0]),int(posstr[1]))
            last_human_action = move_to_take
            board,reward,is_done = game.step(move_to_take)
        elif game.cur_player == -BW:
            AIplayer.observe(game,last_human_action)
            probs=AIplayer.think() #TODO
            move_to_take = AIplayer.take_action()
            print(move_to_take,probs.max())
            board, reward, is_done = game.step(move_to_take)

    print (game)
    if reward == -1:
        winner = "white"
    elif reward == 1:
        winner = "black"
    else:
        winner = "tie"
    print ("Winner is {}".format(winner))


def train(controller,buffer,queue,lock):
    step = 0
    while step < MAX_TRAIN_STEP:
        if (step % 5) == 0:
            try:
                states_a,probs_a,wins_a = queue.get_nowait()
                buffer.append_many(states_a, probs_a, wins_a)
                print ("samples length ",len(states_a))
            except EmptyError:
                pass
        if (len(buffer)<START_TRAIN_BUFFER_SIZE):
            time.sleep(5)
            continue

        step += 1

        sample_states, sample_probs, sample_wins = buffer.sample(BATCH_SIZE)
        with lock:
            loss = controller.train_on_batch(sample_states,sample_wins,sample_probs)
        print (loss)




if __name__=='__main__':
    game = WuZiQi.Env(BOARD_SIZE)
    net = Network.PoliycValueNet(BOARD_SIZE[0], BOARD_SIZE[1], 4)
    buffer = Buffer(BUFFER_SIZE)
    training_controller = Network.Controller(net,lr = 0.001)
    training_controller.model.share_memory()
    lock = mp.Lock()
    queue = mp.Queue(100)


    net = Network.PoliycValueNet(BOARD_SIZE[0], BOARD_SIZE[1], 4)
    playing_controller = Network.Controller(net, lr=LEARNING_RATE)


    AIplayer = MonteCarloTreeSearch.MCTSPlayer(
        playing_controller,C_PUCT,N_SEARCH,
        return_probs=True,temperature=TEMPARETURE,noise=False)

    for i in range(2):

        #self play and add data to queue
        states_a,probs_a,wins_a = self_play(game,AIplayer)
        queue.put([states_a,probs_a,wins_a])

