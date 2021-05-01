# Rainbow RL Algorithm class
# Based on: https://github.com/higgsfield/RL-Adventure

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from training.training import Trainer
from common.replay_buffer import ReplayBuffer
from models.models import RainbowDQN

USE_CUDA = torch.cuda.is_available()
def Variable(x): return x.cuda() if USE_CUDA else x


num_atoms = 51
Vmin = -10
Vmax = 10


class Rainbow(Trainer):

    def __init__(self):
        super(Rainbow, self).__init__()
        self.replay_buffer = ReplayBuffer(self.buffersize)

    def push_to_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def load_model(self):
        self.current_model = RainbowDQN(self.env.observation_space.shape[0],
                                   self.env.action_space.n, num_atoms, Vmin, Vmax)  # input:(1,84,84), output:6
        self.target_model = RainbowDQN(
            self.env.observation_space.shape[0], self.env.action_space.n, num_atoms, Vmin, Vmax)

        if USE_CUDA:
            self.current_model = self.current_model.cuda()
            self.target_model = self.target_model.cuda()

        self.update_target(self.current_model, self.target_model)  # sync nets

    def projection_distribution(self, next_state, rewards, dones):
        batch_size = next_state.size(0)

        delta_z = float(Vmax - Vmin) / (num_atoms - 1)
        support = torch.linspace(Vmin, Vmax, num_atoms)

        next_dist = self.target_model(next_state).data.cpu() * support
        next_action = next_dist.sum(2).max(1)[1]
        next_action = next_action.unsqueeze(1).unsqueeze(
            1).expand(next_dist.size(0), 1, next_dist.size(2))
        next_dist = next_dist.gather(1, next_action).squeeze(1)

        rewards = rewards.unsqueeze(1).expand_as(next_dist)
        dones = dones.unsqueeze(1).expand_as(next_dist)
        support = support.unsqueeze(0).expand_as(next_dist)

        Tz = rewards + (1 - dones) * 0.99 * support
        Tz = Tz.clamp(min=Vmin, max=Vmax)
        b = (Tz - Vmin) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        offset = torch.linspace(0, (batch_size - 1) * num_atoms, batch_size).long()\
            .unsqueeze(1).expand(batch_size, num_atoms)

        proj_dist = torch.zeros(next_dist.size())
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1),
                                    (next_dist * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1),
                                    (next_dist * (b - l.float())).view(-1))

        return proj_dist

    def compute_td_loss(self, batch_size, *args):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = Variable(torch.FloatTensor(np.float32(state)))
        next_state = Variable(torch.FloatTensor(np.float32(next_state)))
        action = Variable(torch.LongTensor(action))
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(np.float32(done))

        proj_dist = self.projection_distribution(next_state, reward, done)

        dist = self.current_model(state)
        action = action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, num_atoms)
        dist = dist.gather(1, action).squeeze(1)
        dist.data.clamp_(0.01, 0.99)
        loss = -(Variable(proj_dist) * dist.log()).sum(1)
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.current_model.reset_noise()
        self.target_model.reset_noise()

        return loss
