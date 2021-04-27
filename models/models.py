import random
import numpy as np

import torch
import torch.nn as nn

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

    def act(self, state, epsilon=1.0):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(np.float32(state))).flatten(0)
            q_value = self.forward(state)
            action = q_value.max(0)[1].item()
        else:
            # negative actions must be caught
            action = - random.randrange(self.num_actions)
        return action
