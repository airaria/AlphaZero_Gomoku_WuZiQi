import WuZiQi
import numpy as np
import MonteCarloTreeSearch
import Network
import copy,os,time
from config import *
from utils import data_augment,Buffer
import torch.multiprocessing as mp
from queue import Empty as EmptyError

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
            posstr= input("action: x,y\n").split(',')
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


def train(controller, buffer, queue, lock):
    step = 0
    while step < MAX_TRAIN_STEP:
        if (step % 5) == 0:
            try:
                states_a, probs_a, wins_a = queue.get_nowait()
                buffer.append_many(states_a, probs_a, wins_a)
                print("samples length ", len(states_a))
            except EmptyError:
                pass
        if (len(buffer) < START_TRAIN_BUFFER_SIZE):
            time.sleep(5)
            continue

        step += 1

        sample_states, sample_probs, sample_wins = buffer.sample(BATCH_SIZE)
        with lock:
            loss = controller.train_on_batch(sample_states, sample_wins, sample_probs)
        print(loss)


def train_epoch(controller, buffer, queue, lock, barrier, done_event, save_dir):

    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    while not done_event.is_set():
        count = 0
        print ("training process is waiting...")
        barrier.wait()
        print ("training process has crossed the barrier.")
        time.sleep(1)
        while True:
            try:
                states_a, probs_a, wins_a = queue.get_nowait()
                buffer.append_many(states_a, probs_a, wins_a)
                print("samples length ", len(states_a))
            except EmptyError:
                break

        if (len(buffer) < START_TRAIN_BUFFER_SIZE):
            print ("Buffer size {} is still too small, waiting for more ".format(len(buffer)))
        else:
            #TODO train loop
            loss = 0
            NoB = N_EPOCH_PER_TRAIN_STEP*buffer.num_sample//BATCH_SIZE
            print ("Number of batches: ",NoB)
            for i_batch in range(NoB):
                sample_states, sample_probs, sample_wins = buffer.sample(BATCH_SIZE)
                '''
                with lock: #time consuming?
                    loss += controller.train_on_batch(sample_states, sample_wins, sample_probs)
                '''
                # Sequential training , don't need lock
                loss += controller.train_on_batch(sample_states, sample_wins, sample_probs)

            print("Average loss: ",loss/NoB)
            print ("A training iteration is finished, start next self-play")

        count += 1
        if count%10==0:
            training_controller.save2file(os.path.join(save_dir,"model_{:05d}.pkl".format(count)))

        barrier.wait()
        time.sleep(1)

    print ("Detect done_event. Training is ended.")

def collect_self_play_data(game,queue,lock,barrier,done_event,
                           num_self_play=MAX_SELF_PLAY,
                           training_model=None):

    net = Network.PoliycValueNet(BOARD_SIZE[0], BOARD_SIZE[1], 4)
    playing_controller = Network.Controller(net, lr=LEARNING_RATE)


    AIplayer = MonteCarloTreeSearch.MCTSPlayer(
        playing_controller,C_PUCT,N_SEARCH,
        return_probs=True,temperature=TEMPARETURE,noise=True)

    for i in range(num_self_play):
        #load neweset net state
        if not training_model is None:
            '''
            with lock:
                state_dict = training_model.state_dict()
                AIplayer.controller.model.load_state_dict(state_dict)
            '''
            # Sequential training , don't need lock
            state_dict = training_model.state_dict()
            AIplayer.controller.model.load_state_dict(state_dict)

        #self play and add data to queue
        states_a,probs_a,wins_a = self_play(game,AIplayer)
        queue.put([states_a,probs_a,wins_a])

        print ("{}-th play generated, waiting for passing barrier".format(i))
        barrier.wait()
        time.sleep(0.5)
        barrier.wait()

    print ("Set up done_event.")
    done_event.set()
    print ("Self-play finished.")

if __name__=='__main__':

    game = WuZiQi.Env(BOARD_SIZE)
    net = Network.PoliycValueNet(BOARD_SIZE[0], BOARD_SIZE[1], 4)

    if MODE == 'TRAIN':
        ctx = mp.get_context('forkserver')
        buffer = Buffer(BUFFER_SIZE)
        training_controller = Network.Controller(net,lr = LEARNING_RATE)
        training_controller.model.share_memory()

        m = mp.Manager()
        lock = m.Lock()
        queue = m.Queue(50)
        barrier = m.Barrier(N_WORKER+1)
        done_event = m.Event()

        workers = []
        for i in range(N_WORKER):
            worker = ctx.Process(target = collect_self_play_data,
                                args=(game,queue,lock,barrier,done_event,
                                      MAX_SELF_PLAY,training_controller.model))
            workers.append(worker)
            worker.start()

        #Training is much faster than self-paly. So there is no need to be async.
        train_epoch(training_controller, buffer, queue, lock, barrier, done_event, SAVE_DIR)

        for worker in workers:
            worker.join()


    if MODE == 'TEST':
        test_controller = Network.Controller(net,lr = LEARNING_RATE)
        #test_controller.load_file(LOAD_FN)
        AIplayer = MonteCarloTreeSearch.MCTSPlayer(
            test_controller, C_PUCT, N_SEARCH,
            return_probs=True, temperature=TEMPARETURE, noise=False)
        human_play(game, AIplayer, BLACK)
