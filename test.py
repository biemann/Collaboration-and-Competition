from unityagents import UnityEnvironment

from maddpg import MADDPG
from ddpg import ReplayBuffer
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque


env = UnityEnvironment(file_name = "Tennis.app")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]


agent = MADDPG(discount_factor = 0.99, tau = 0.02, batch_size = 256)
agent.maddpg_agent[0].actor.load_state_dict(torch.load('bin/actor0_finished.pth', map_location = lambda storage, loc: storage))
agent.maddpg_agent[1].actor.load_state_dict(torch.load('bin/actor1_finished.pth', map_location = lambda storage, loc: storage))

env_info = env.reset(train_mode = False)[brain_name]       
state = env_info.vector_observations 
state =  torch.from_numpy(np.array(state)).float().unsqueeze(0)
score = np.zeros(2) 

while True:
    actions = agent.act(state, 0)
    actions_array = torch.stack(actions).detach().numpy()
    env_info = env.step(actions_array)[brain_name]  
    next_state = env_info.vector_observations         
    next_state = torch.from_numpy(np.array(next_state)).float().unsqueeze(0)
    reward = np.array(env_info.rewards).reshape(1, -1)
    dones = np.array(env_info.local_done).reshape(1, -1)   
    actions_array = actions_array.reshape(1, -1)               
					
    score += reward[0]		                         
    state = next_state                             
    if np.any(dones):                                
        break


print(np.max(score))
env.close()