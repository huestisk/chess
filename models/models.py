import random
import numpy as np

import torch
import torch.nn as nn

from common.layers import NoisyLinear

USE_CUDA = torch.cuda.is_available()
def Variable(x): return x.cuda() if USE_CUDA else x


class ChessNN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(ChessNN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions

        self.layers = nn.Sequential(		# TODO
            nn.Linear(np.prod(input_shape), 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon=0.0):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(np.float32(state))).flatten(0)
            q_value = self.forward(state)
            action = q_value.max(0)[1].item()
        else:
            # negative actions must be caught
            action = - random.randrange(self.num_actions)
        return action



class RainbowCnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions, num_atoms, Vmin, Vmax):
        super(RainbowCnnDQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.Vmin = Vmin
        self.Vmax = Vmax

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.noisy_value1 = NoisyLinear(
            self.feature_size(), 512, use_cuda=USE_CUDA)
        self.noisy_value2 = NoisyLinear(512, self.num_atoms, use_cuda=USE_CUDA)

        self.noisy_advantage1 = NoisyLinear(
            self.feature_size(), 512, use_cuda=USE_CUDA)
        self.noisy_advantage2 = NoisyLinear(
            512, self.num_atoms * self.num_actions, use_cuda=USE_CUDA)

    def forward(self, x):
        batch_size = x.size(0)

        x = x / 255.
        x = self.features(x)
        x = x.view(batch_size, -1)

        value = F.relu(self.noisy_value1(x))
        value = self.noisy_value2(value)

        advantage = F.relu(self.noisy_advantage1(x))
        advantage = self.noisy_advantage2(advantage)

        value = value.view(batch_size, 1, self.num_atoms)
        advantage = advantage.view(
            batch_size, self.num_actions, self.num_atoms)

        x = value + advantage - advantage.mean(1, keepdim=True)
        x = F.softmax(x.view(-1, self.num_atoms), dim=-
                      1).view(-1, self.num_actions, self.num_atoms)

        return x

    def reset_noise(self):
        self.noisy_value1.reset_noise()
        self.noisy_value2.reset_noise()
        self.noisy_advantage1.reset_noise()
        self.noisy_advantage2.reset_noise()

    def feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)

    def act(self, state):
        state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0))
        dist = self.forward(state).data.cpu()
        dist = dist * torch.linspace(self.Vmin, self.Vmax, self.num_atoms)
        action = dist.sum(2).max(1)[1].numpy()[0]
        return action

