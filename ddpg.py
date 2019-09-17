# individual network settings for each actor + critic pair
# see networkforall for details

from model import Network
import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Variable
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch
import numpy as np
from collections import deque
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPGAgent:
    def __init__(self, in_actor, hidden_in_actor, hidden_out_actor, out_actor, 
                in_critic, hidden_in_critic, hidden_out_critic, lr_actor = 5e-4, lr_critic = 5e-4):
        super(DDPGAgent, self).__init__()

        self.actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor = True).to(device)
        self.critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)
        self.target_actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor = True).to(device)
        self.target_critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)

        self.noise = OUNoise(out_actor, scale = 1.0)
        
        self.actor_optimizer = Adam(self.actor.parameters(), lr = lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr = lr_critic, weight_decay = 0)


    def act(self, state, noise):
        state = state.unsqueeze(0).to(device)
        self.actor.eval()
        action = self.actor(state) + noise * self.noise.noise()
        return action[0]

    def target_act(self, state, noise):
        state = state.to(device)
        action = self.target_actor(state) + noise * self.noise.noise()
        return action



class OUNoise:

    def __init__(self, action_dimension, scale = 1.0, mu = 0, theta = 0.15, sigma = 0.05):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return torch.tensor(self.state * self.scale).float()



def transpose_list(mylist):
    return list(map(list, zip(*mylist)))


class ReplayBuffer:
    def __init__(self,size):
        self.size = size
        self.deque = deque(maxlen = self.size)

    def push(self,transition):
        """push into the buffer"""
        
        input_to_buffer = transpose_list(transition)
    
        for item in input_to_buffer:
            self.deque.append(item)

    def sample(self, batchsize):
        """sample from the buffer"""
        samples = random.sample(self.deque, batchsize)

        # transpose list of list
        return transpose_list(samples)

    def __len__(self):
        return len(self.deque)



