EPOCH = 400
BATCH_SIZE = 128
LR = 0.005                   # 学习率
EPSILON = 0.9            # 最优选择动作百分比(有0.9的几率是最大选择，还有0.1是随机选择，增加网络能学到的Q值)
GAMMA = 0.8                 # 奖励递减参数（衰减作用，如果没有奖励值r=0，则衰减Q值）
TARGET_REPLACE_ITER = 100   # Q 现实网络的更新频率100次循环更新一次
MEMORY_CAPACITY = 1600      # 记忆库大小
N_ACTIONS = 4  # 棋子的动作0，1，2，3
N_STATES = 1