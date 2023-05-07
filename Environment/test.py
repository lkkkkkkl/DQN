import numpy as np
import pickle 
import torch
import copy
def load():
    with open(r'D:\GitHub\DQN\data_saved\position_001.pickle', 'rb') as file:
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

listz = load()
#for i in range(len(listz)):

print(len(listz))
#for i in range(10):
#print(listz[0][0])
#print(torch.FloatTensor(a_dict1[0][0]))

