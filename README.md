# AlphaZero_Gomoku_WuZiQi

##  config.py 配置

一些参数说明：

- BOARD_SIZE : 棋盘大小。
- C_PUCT : 探索程度控制，参见AlphaGo Zero原文。
- N_SEARCH ：每走一步执行多少次局面求值。
- TEMPERATURE：走子随机化程度控制，参见AlphaGo Zero原文。
- MAX_SELF_PLAY：每个进程的总对局数。
- N_VIRTUAL_LOSS：virtual_loss大小，参见AlphaGo原文。
- N_EVALUATE：每次探索N_EVALUATE个叶子节点，再统一计算这些节点的value。此值>1时，请保证N_VIRTUAL_LOSS > 0。
- N_WORKER：并行self-play的进程数。**每步的探索次数=N_WORKER * N_SEARCH * N_EVALUATE** 。
- BUFFER_SIZE：储存对弈局面的buffer大小。
- N_EPOCH_PER_TRAIN_STEP：每次训练时，在已收集的self-play数据上过几个epoch。
- SELF_PLAY_PER_TRAIN：每次训练前，每个进程self-play的局数
- SAVE_EVERY_N_EPOCH：顾名思义。
- START_TRAIN_BUFFER_SIZE：buffer中至少收集到这么多数据时才开始训练。
- SAVE_DIR：模型保存目录
- LOAD_FN：待加载模型文件，值为文件名或None。
- MODE：
  - 等于 "TRAIN" 时：如果LOAD_FN不是None ，那么加载模型文件并训练；否则从头训练。
  - 等于 "TEST" 时，如果LOAD_FN不是None，那么加载模型文件并进入人机对战模式；否则随机初始化模型并对战。
- MAX_TO_KEEP：最多保存多少个模型文件。

## 编译动态库

gcc -shared -fPIC  arb.c -o libarb.so

## 训练

python main.py

## 人机对战

python main.py  

输入格式为 

x,y

## 已知问题

1. 尚未实现自我对弈的模型评测