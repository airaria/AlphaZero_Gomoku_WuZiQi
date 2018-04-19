import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import WuZiQi
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

        #print(game)
        AIplayer.observe(game)
        probs = AIplayer.think()

        #print(probs.max(), probs.argmax() // BOARD_SIZE[0], probs.argmax() % BOARD_SIZE[1])
        move_to_take = AIplayer.take_action()
        #print (move_to_take)

        states_list.append(game.build_features(rot_and_flip=False))
        probs_list.append(probs)
        current_player_list.append(game.cur_player)

        _, reward, is_done = game.step(move_to_take)
    #print ("done")
    #print(game)
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

            before_value = (AIplayer.mcts.root.leaf_value+1)/2

            move_to_take = AIplayer.take_action()

            print(move_to_take)
            print ("Move probablity:",probs.max())
            print ("Current player {}\nWin probability(before):{}".format(
                game.cur_player,before_value))

            board, reward, is_done = game.step(move_to_take)

            after_value = (AIplayer.mcts.root.leaf_value+1)/2
            print("Current player {}\nWin probability(after):{}".format(
                game.cur_player,after_value))

    print (game)
    if reward == -1:
        winner = "white"
    elif reward == 1:
        winner = "black"
    else:
        winner = "tie"
    print ("Winner is {}".format(winner))

def self_eval(game,p1,p2):
    game = copy.deepcopy(game)
    game.reset()
    p1.reset()
    p2.reset()
    is_done = False
    last_oppsite_action = None
    while not is_done:
        if game.cur_player == BLACK:
            print ("black thinking")
            p1.observe(game,last_oppsite_action)
            probs = p1.think()
            move_to_take = p1.take_action()
            last_oppsite_action = move_to_take
            _, before_value = p1.mcts.value_fn(game)

            print(move_to_take)
            print ("Current player {}\nWin probability(before):{}".format(
                game.cur_player,(before_value+1)/2))

            board, reward, is_done = game.step(move_to_take)

            print (game)
            _, after_value = p1.mcts.value_fn(game)
            print("Current player {}\nWin probability(after):{}".format(
                game.cur_player,(after_value+1)/2))

        else:
            print ("white thinking")
            p2.observe(game,last_oppsite_action)
            probs = p2.think()
            move_to_take = p2.take_action()
            last_oppsite_action = move_to_take
            _, before_value = p2.mcts.value_fn(game)

            print ("Current player {}\nWin probability(before):{}".format(
                game.cur_player,(before_value+1)/2))

            board, reward, is_done = game.step(move_to_take)

            print (game)
            _, after_value = p2.mcts.value_fn(game)
            print("Current player {}\nWin probability(after):{}".format(
                game.cur_player,(after_value+1)/2))

    if reward == -1:
        winner = "white"
    elif reward == 1:
        winner = "black"
    else:
        winner = "tie"
    print ("Winner is {}".format(winner))


'''
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
'''

def train_epoch(controller, buffer, queue, lock, barrier, done_event, save_dir):

    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    count = 0

    while not done_event.is_set():
        print ("training process is waiting...")
        barrier.wait()
        print ("training process has crossed the barrier.")
        time.sleep(1)

        n_new_sample = 0
        while True:
            try:
                states_a, probs_a, wins_a = queue.get_nowait()
                buffer.append_many(states_a, probs_a, wins_a)
                n_new_sample += len(states_a)
                print("samples length ", len(states_a))
            except EmptyError:
                break

        if (len(buffer) < START_TRAIN_BUFFER_SIZE):
            print ("Buffer size {} is still too small, waiting for more ".format(len(buffer)))
        elif n_new_sample==0:
            print ("No new data.")
        else:
            #train loop
            loss = 0
            NoB = int(N_EPOCH_PER_TRAIN_STEP*buffer.num_sample//BATCH_SIZE)
            print ("Number of sample: ",buffer.num_sample)
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
            print ("{}-th iteration is finished, start next self-play".format(count+1))
            count += 1

        if count%SAVE_EVERY_N_EPOCH==0:
            print ("saving %d"% count)
            training_controller.save2file(
                os.path.join(save_dir,"model_{:05d}.pkl".format(count)),MAX_TO_KEEP)

        barrier.wait()
        time.sleep(1)

    print ("Detect done_event. Training is finished.")


def collect_self_play_data(game,queue,lock,barrier,done_event,
                           num_self_play=MAX_SELF_PLAY,
                           training_model=None):

    net = Network.PoliycValueNet(BOARD_SIZE[0], BOARD_SIZE[1], 4)
    playing_controller = Network.Controller(net, lr=LEARNING_RATE,L2_weight=L2_WEIGHT)


    AIplayer = MonteCarloTreeSearch.MCTSPlayer(
        playing_controller,C_PUCT,N_SEARCH,
        return_probs=True,temperature=TEMPERATURE,noise=True,dir_noise=True)

    for i in range(num_self_play):
        print ("Start %d-th self-play" % (i+1))
        #load neweset net state
        if not training_model is None:
            '''
            with lock:
                state_dict = training_model.state_dict()
                AIplayer.controller.model.load_state_dict(state_dict)
            '''
            # Sequential self-play and training , don't need lock
            state_dict = training_model.state_dict()
            AIplayer.controller.model.load_state_dict(state_dict)

        #self play and add data to queue
        states_a,probs_a,wins_a = self_play(game,AIplayer)
        queue.put([states_a,probs_a,wins_a])

        if (i+1) % SELY_PLAY_PER_TRAIN == 0:
            print ("waiting for passing barrier".format(i+1))
            barrier.wait()
            time.sleep(0.5)
            barrier.wait()

    print("waiting for passing barrier".format(i + 1))
    barrier.wait()
    time.sleep(0.5)
    barrier.wait()

    print ("Set up done_event.")
    done_event.set()
    print ("Self-play finished.")

if __name__=='__main__':

    game = WuZiQi.Env(BOARD_SIZE)


    if MODE == 'TRAIN':
        ctx = mp.get_context('forkserver')
        buffer = Buffer(BUFFER_SIZE)
        net = Network.PoliycValueNet(BOARD_SIZE[0], BOARD_SIZE[1], 4)
        training_controller = Network.Controller(net,lr = LEARNING_RATE,L2_weight=L2_WEIGHT)

        if not LOAD_FN is None:
            training_controller.load_file(LOAD_FN)

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
        net = Network.PoliycValueNet(BOARD_SIZE[0], BOARD_SIZE[1], 4)
        test_controller = Network.Controller(net,lr = LEARNING_RATE,L2_weight=L2_WEIGHT)
        if not LOAD_FN is None:
            test_controller.load_file(LOAD_FN)
        AIplayer = MonteCarloTreeSearch.MCTSPlayer(
            test_controller, C_PUCT, N_SEARCH,
            return_probs=True, temperature=TEMPERATURE, noise=False,dir_noise=False)
        human_play(game, AIplayer, WHITE)

    if MODE == 'EVAL':
        net = Network.PoliycValueNet(BOARD_SIZE[0], BOARD_SIZE[1], 4)
        P1_controller = Network.Controller(net,lr = LEARNING_RATE,L2_weight=L2_WEIGHT)
        net = Network.PoliycValueNet(BOARD_SIZE[0], BOARD_SIZE[1], 4)
        P2_controller = Network.Controller(net,lr = LEARNING_RATE,L2_weight=L2_WEIGHT)

        P1_controller.load_file(P1)
        p1 = MonteCarloTreeSearch.MCTSPlayer(
            P1_controller,C_PUCT,N_SEARCH,
            return_probs=True, temperature=TEMPERATURE,noise=False,dir_noise=True)

        P2_controller.load_file(P2)
        p2 = MonteCarloTreeSearch.MCTSPlayer(
            P2_controller,C_PUCT,N_SEARCH,
            return_probs=True, temperature=TEMPERATURE,noise=False,dir_noise=True)

        self_eval(game,p1,p2)
