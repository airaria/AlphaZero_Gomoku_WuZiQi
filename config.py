BOARD_SIZE = (11,11)
BLACK = 1
WHITE = -1
C_PUCT = 2
N_SEARCH = 850 #700
TEMPERATURE = 1.0
L2_WEIGHT = 1e-4

BATCH_SIZE = 128
MAX_SELF_PLAY = 1000
LEARNING_RATE = 0.0005
N_VIRTUAL_LOSS = 4
N_EVALUATE = 2#3
N_WORKER = 6

BUFFER_SIZE = 30000
N_EPOCH_PER_TRAIN_STEP = 0.5
SELY_PLAY_PER_TRAIN = 2
SAVE_EVERY_N_EPOCH = 5
START_TRAIN_BUFFER_SIZE = 20480

SAVE_DIR = 'saved_model_6th_experiment/'
LOAD_FN =  'saved_model_5th_experiment/model_00050.pkl'
MODE = "TEST" # 'TRAIN' or "TEST" or "EVAL"
MAX_TO_KEEP = 20


P1 = 'saved_model_4th_experiment/model_00250.pkl'
P2 = 'saved_model_4th_experiment/model_00250.pkl'
