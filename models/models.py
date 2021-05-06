import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.layers import NoisyLinear

USE_CUDA = torch.cuda.is_available()
def Variable(x): return x.cuda() if USE_CUDA else x


class ChessNN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(ChessNN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions

        # self.layers = nn.Sequential(
        #     nn.Linear(np.prod(input_shape), 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, self.num_actions)
        # )

        self.layers = nn.Sequential(
            nn.Linear(np.prod(input_shape), self.num_actions)
        )

    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon=0.0):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(np.float32(state))).flatten(0)
            q_value = self.forward(state)
            action = q_value.max(0)[1].item()
        else:
            # negative SHIFTED actions must be caught
            action = - random.randrange(self.num_actions) - 1
        return action


class RainbowDQN(nn.Module):
    def __init__(self, num_inputs, num_actions, num_atoms, Vmin, Vmax):
        super(RainbowDQN, self).__init__()

        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.Vmin = Vmin
        self.Vmax = Vmax

        self.linear1 = nn.Linear(num_inputs, 32)
        self.linear2 = nn.Linear(32, 64)

        self.noisy_value1 = NoisyLinear(64, 64, use_cuda=USE_CUDA)
        self.noisy_value2 = NoisyLinear(64, self.num_atoms, use_cuda=USE_CUDA)

        self.noisy_advantage1 = NoisyLinear(64, 64, use_cuda=USE_CUDA)
        self.noisy_advantage2 = NoisyLinear(
            64, self.num_atoms * self.num_actions, use_cuda=USE_CUDA)

    def forward(self, x):
        batch_size = x.size(0)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        value = F.relu(self.noisy_value1(x))
        value = self.noisy_value2(value)

        advantage = F.relu(self.noisy_advantage1(x))
        advantage = self.noisy_advantage2(advantage)

        value = value.view(batch_size, 1, self.num_atoms)
        advantage = advantage.view(
            batch_size, self.num_actions, self.num_atoms)

        x = value + advantage - advantage.mean(1, keepdim=True)
        x = F.softmax(x.view(-1, self.num_atoms),dim=1).view(-1,self.num_actions, self.num_atoms)

        return x

    def reset_noise(self):
        self.noisy_value1.reset_noise()
        self.noisy_value2.reset_noise()
        self.noisy_advantage1.reset_noise()
        self.noisy_advantage2.reset_noise()

    def act(self, state, epsilon=0.0):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(state).unsqueeze(0))
            dist = self.forward(state).data.cpu()
            dist = dist * torch.linspace(self.Vmin, self.Vmax, self.num_atoms)
            action = dist.sum(2).max(1)[1].numpy()[0]
        else:
            action = - random.randrange(self.num_actions)
        return action
