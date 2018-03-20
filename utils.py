import numpy as np
from config import *

def data_augment(states_list,probs_list,wins_list):
    states = []
    probs = []
    for rot in range(4):
        rotated_states_list = [np.rot90(f, rot, axes=(1, 2)) for f in states_list]
        flipped_states_list = [np.flip(f,axis=-1) for f in rotated_states_list]
        rotated_probs_list =  [np.rot90(f.reshape(BOARD_SIZE), rot).reshape(-1) for f in probs_list]
        flipped_probs_list =  [np.flip(f.reshape(BOARD_SIZE),axis=-1).reshape(-1) for f in rotated_probs_list]

        states += rotated_states_list
        states += flipped_states_list

        probs += rotated_probs_list
        probs += flipped_probs_list

    return np.array(states,dtype=np.float32,),\
           np.array(probs,dtype=np.float32),\
           np.array(wins_list*8,dtype=np.float32).reshape(-1,1)


class Buffer(object):
    def __init__(self,buffer_size):
        self.state_buffer = np.zeros((buffer_size, 4,BOARD_SIZE[0], BOARD_SIZE[1]), dtype=np.float32)
        self.prob_buffer = np.zeros((buffer_size, BOARD_SIZE[0] * BOARD_SIZE[1]), dtype=np.float32)
        self.win_buffer = np.zeros((buffer_size, 1), dtype=np.float32)
        self.buffer_index = 0
        self.num_sample = 0
        self.buffer_size = buffer_size
    def append_many(self,state_a,prob_a,win_a):
        print (state_a.shape,prob_a.shape,win_a.shape)
        data_length = state_a.shape[0]
        if self.buffer_index + data_length < self.buffer_size:
            b, e = self.buffer_index, self.buffer_index + data_length
            self.state_buffer[b:e] = state_a
            self.prob_buffer[b:e] = prob_a
            self.win_buffer[b:e] = win_a
            self.buffer_index = (self.buffer_index + data_length) % self.buffer_size
            self.num_sample += data_length
        else:
            batch_1_size = self.buffer_size - self.buffer_index
            batch_2_size = data_length - batch_1_size

            print ("batch_1_size",batch_1_size)
            print ("batch_2_size",batch_2_size)

            b = self.buffer_index
            self.state_buffer[b:] = state_a[:batch_1_size]
            self.prob_buffer[b:] = prob_a[:batch_1_size]
            self.win_buffer[b:] = win_a[:batch_1_size]
            self.state_buffer[:batch_2_size] = state_a[batch_1_size:]
            self.prob_buffer[:batch_2_size] = prob_a[batch_1_size:]
            self.win_buffer[:batch_2_size] = win_a[batch_1_size:]
            self.buffer_index = batch_2_size
            self.num_sample += data_length

        self.num_sample = min(self.buffer_size, self.num_sample)
    def sample(self,batch_size):
        index = np.random.choice(self.num_sample,batch_size)
        sample_states = self.state_buffer[index]
        sample_probs  = self.prob_buffer[index]
        sample_wins   = self.win_buffer[index]

        return sample_states,sample_probs,sample_wins

    def __len__(self):
        return self.num_sample