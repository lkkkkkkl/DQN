from Environment import Env
import PARAMETER
import numpy as np
import pickle
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def seed_torch(seed=2023):
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

"""def load_fromnpy(filepath):
    data_npy = np.load(filepath, allow_pickle=True)
    return data_npy"""
def load_frompickle(filepath):
    with open(filepath, 'rb') as file:
        dict = []
        while True:
            try:
                a_dict1 = pickle.load(file)
                dict.append(a_dict1)
            except:
                dic = []
                for i in range(0,len(dict)):
                    for j in range(len(dict[i])):
                        dic.append(dict[i][j])
                return dic



def trans_torch(list1):
    #print('list1 ',list1)
    list1 = np.array(list1)
    l1 = np.where(list1 == 1, 1, 0)
    l2 = np.where(list1 == 2, 1, 0)
    l3 = np.where(list1 == 3, 1, 0)
    #print('l1:',l1,'l2:',l2,'l3:',l3)
    b = np.array([l1, l2, l3])
    #print('b',b)
    return b


# 神经网络
class Net(nn.Module):
    def __init__(self, num_input):
        super(Net, self).__init__()
        self.c1 = nn.Conv2d(3, 25, 5, 1, 0)
        self.f0 = nn.Linear(num_input, 256)
        self.f1 = nn.Linear(256, 32)
        self.f1.weight.data.normal_(0, 0.1)
        self.f2 = nn.Linear(32, 4)
        self.f2.weight.data.normal_(0, 0.1)


    def forward(self, x):
        x = x.view(x.size(0), -1)
        # x = self.c1(x)
        x = self.f0(x)
        x = F.relu(x)
        # x = x.view(x.size(0), -1)
        x = self.f1(x)
        x = F.relu(x)
        action = self.f2(x)
        return action


class DQN(object):
    def __init__(self,num_input):
        self.eval_net, self.target_net = Net(num_input), Net(num_input)  # DQN需要使用两个神经网络
        # eval为Q估计神经网络 target为Q现实神经网络
        self.learn_step_counter = 0  # 用于 target 更新计时，50次更新一次
        self.memory_counter = 0  # 记忆库记数
        self.MEMORY_ALL = load_frompickle(r'D:\GitHub\DQN\data_saved\position_001.pickle')
        self.batch_idx = 0
        #self.memory = list(np.zeros((PARAMETER.MEMORY_CAPACITY, 4)))  # 初始化记忆库用numpy生成一个(容量,4)大小的全0矩阵
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=PARAMETER.LR)  # torch 的优化器
        self.loss_func = nn.MSELoss()   # 误差公式
        self.reward_best = -10

    def choose_action(self, x, epsilon):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)  # ??
        # 这里只输入一个 sample, x为场景
        if np.random.uniform() < epsilon:   # 选最优动作
            actions_value = self.eval_net.forward(x)  # 将场景输入Q估计神经网络
            # torch.max(input,dim)返回dim最大值并且在第二个位置返回位置比如(tensor([0.6507]), tensor([2]))
            action = torch.max(actions_value, 1)[1].data.numpy()  # 返回动作最大值
        else:   # 选随机动作
            action = np.array([np.random.randint(0, PARAMETER.N_ACTIONS)])  # 比如np.random.randint(0,2)是选择1或0
        return action

    def store_reset(self):
        # 如果记忆库满了, 就覆盖老数据，2000次覆盖一次
        index = self.memory_counter % PARAMETER.MEMORY_CAPACITY
        if index == 0 and self.memory_counter != 0:
            self.batch_idx += PARAMETER.MEMORY_CAPACITY
            return 1
        self.memory_counter += 1
        print(self.memory_counter, self.batch_idx)
        #quits = 0
        return 0
        


    def learn(self):
        # target net 参数更新,每100次
        if self.learn_step_counter % PARAMETER.TARGET_REPLACE_ITER == 0:
            # 将所有的eval_net里面的参数复制到target_net里面
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        # 抽取记忆库中的批数据
        # 从2000以内选择32个数据标签
        sample_index = np.random.choice(PARAMETER.MEMORY_CAPACITY, PARAMETER.BATCH_SIZE)
        b_s  = []
        b_a  = []
        b_r  = []
        b_s_ = []
        #print(self.MEMORY_ALL[1][0], self.memory[2][0])
        for i in sample_index:
            b_s.append(self.MEMORY_ALL[i+self.batch_idx][0])
            b_a.append(np.array(self.MEMORY_ALL[i+self.batch_idx][1], dtype=np.int32))
            #b_a.append(int(float(self.memory[i][1])))
            b_r.append(np.array([self.MEMORY_ALL[i+self.batch_idx][2]], dtype=np.int32))
            b_s_.append(self.MEMORY_ALL[i+self.batch_idx][3])
        #print(b_s)
        b_s = torch.FloatTensor(b_s)  # 取出s
        b_a = torch.LongTensor(b_a)  # 取出a
        b_r = torch.FloatTensor(b_r)  # 取出r
        b_s_ = torch.FloatTensor(b_s_)  # 取出s_
        #print(b_s.size(), b_a.size())
        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        # t = self.eval_net(b_s) 个人感觉，gather使得代码更加优雅。实际是一个从t中进行索引的东西。
        # gather 是按照index拿出目标索引的函数，第一个输入为dim.
        # gather 对应了argmax a. DQN是off-policy的。 b_r + q_next - 1_eval
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1) 找到action的Q估计(关于gather使用下面有介绍)
        q_next = self.target_net(b_s_).detach()     # q_next 不进行反向传递误差, 所以 detach Q现实
        # p = q_next.max(1)[0]
        # b_r: nsamples*1 q_next.max(1)[0]:nsamples
        # q_eval: nsamples*1
        q_target = b_r + PARAMETER.GAMMA * torch.unsqueeze(q_next.max(1)[0], 1)  # shape (batch, 1) DQL核心公式
        # 这步走的不好，将导致下一次判断做大调整。
        loss = self.loss_func(q_eval, q_target)  # 计算误差
        # 计算, 更新 eval net
        self.optimizer.zero_grad()
        loss.backward()  # 反向传递
        self.optimizer.step()


def main():
    dict_a = {0:'up',1:'down',2:'left',3:'right'}
    env = Env()
    dqn = DQN(env.total_grid)  # 定义 DQN 系统
    # 400步
    epsilon = PARAMETER.EPSILON
    for i_episode in range(PARAMETER.EPOCH):
        s = env.start_env()
        s = trans_torch(s)
        his_step = np.zeros_like(s[0, :])
        if epsilon % 100 == 0:
            epsilon += 0.02
        print('eposide {}'.format(i_episode))
        while True:
            a = dqn.choose_action(s, epsilon)  # 选择动作
            #a_lib.append(dict_a[int(a)])
            # 选动作, 得到环境反馈
            done, r, s_ = env.step(a, his_step)  # done 表示是否结束。其中掉入陷阱'1'或者走出迷宫'2'都算结束
            #print(s_)
            s_ = trans_torch(s_)
            this_step = np.where(np.array(s_[0,:]) == 1, 1, 0)
            his_step = np.where((his_step + this_step) >= 1, 1, 0)
            #print(his_step,"\n")
            # 存记忆 好的坏的都要存储，不然网络不能学习完整（比如不知道终点就在附近了）
            if dqn.memory_counter > PARAMETER.MEMORY_CAPACITY:
                dqn.learn()  # 记忆库满了就进行学习
            if done == 1 or done == 2:    # 如果回合结束, 进入下回合
                if done == 1:
                    print('epoch', i_episode, '失败')
                    break
                if done == 2:
                    print('epoch', i_episode, '成功')
                    break

if __name__ == '__main__':
    seed_torch(2023)
    main()