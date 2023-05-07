import cv2
import numpy as np
import random
import time
import copy

class Env:
    def __init__(self):
        super(Env, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.migong = []
        self.x1, self.y1 = 0, 0
        self.end_game = 0
        self.display1 = None
        self.last_action = None
        self.total_grid = len(self.start_env())*len(self.start_env()[0])*3

    # 建立虚拟环境
    def start_env(self):

        """
        self.migong = [[1, 0, 0, 0, 0],
                       [0, 0, 0, 3, 0],
                       [0, 0, 0, 0, 0],
                       [0, 3, 0, 0, 0],
                       [0, 0, 0, 0, 2]]
        """
        self.migong = [[0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2],# x-1对应左，l
                       [0, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 2],# y-1对应上，u
                       [0, 3, 0, 3, 0, 0, 3, 3, 3, 3, 3, 2],# x+1对应右，r
                       [0, 3, 0, 3, 0, 0, 3, 3, 3, 0, 3, 2],# y+1对应下，d
                       [0, 3, 0, 3, 0, 0, 0, 3, 0, 0, 0, 2],
                       [0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 2],
                       [0, 3, 0, 3, 0, 3, 0, 0, 0, 3, 0, 2],
                       [0, 0, 0, 0, 0, 3, 0, 3, 0, 3, 0, 2],
                       [0, 3, 3, 0, 0, 3, 0, 3, 3, 3, 0, 2],
                       [0, 3, 3, 0, 0, 3, 0, 3, 3, 3, 0, 2],
                       [0, 3, 3, 0, 0, 3, 0, 3, 3, 3, 3, 2],]# x+是向右，y+是向下

        self.height = len(self.migong)
        self.width = len(self.migong[0])
        self.x1, self.y1 = 0,int(self.height/2)
        self.migong[self.y1][self.x1] = 1
        self.raw_mg = copy.deepcopy(self.migong)
        self.end_game = 0

        return self.migong

    def assign_reward(self, r, his_step):
        if his_step[self.y1][self.x1] == 1:
            r -= 5
        else: r -= 1
        if self.raw_mg[self.y1][self.x1] == 3:
            #self.end_game = 1
            r += -2
        elif self.raw_mg[self.y1][self.x1] == 2:
            self.end_game = 2
            r += 20
        return r

    def step(self, action, his_step):
        r = 0
        # ['u'0, 'd'1, 'l'2, 'r'3]
        # if (self.last_action, action) in [(0, 1), (1, 0), (2, 3), (3,2)]:
        #    r += -0.5
        if action == 0:
            if self.y1 == 0:
                r += -10
                self.end_game = 1
            else:
                self.migong[self.y1][self.x1] = 0
                self.migong[self.y1 - 1][self.x1] = 1
                self.y1 -= 1
                r = self.assign_reward(r, his_step)

        if action == 1:
            if self.y1 == self.height-1:
                r += -10
                self.end_game = 1
            else:
                self.migong[self.y1][self.x1] = 0
                self.migong[self.y1 + 1][self.x1] = 1
                self.y1 += 1
                r = self.assign_reward(r, his_step)
        if action == 2:
            if self.x1 == 0:
                r += -10
                self.end_game = 1
            else:
                self.migong[self.y1][self.x1] = 0
                self.migong[self.y1][self.x1 - 1] = 1
                self.x1 -= 1
                r = self.assign_reward(r, his_step)
        if action == 3:
            if self.x1 == self.width-1:
                r += -10
                self.end_game = 1
            else:
                self.migong[self.y1][self.x1] = 0
                self.migong[self.y1][self.x1 + 1] = 1
                self.x1 += 1
                r = self.assign_reward(r, his_step)
        # return self.migong
        return self.end_game, r, self.migong
