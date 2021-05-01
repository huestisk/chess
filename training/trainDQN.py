import torch
import numpy as np
from training.training import Trainer
from common.replay_buffer import ReplayBuffer


# Use GPU, if available
USE_CUDA = torch.cuda.is_available()
def Variable(x): return x.cuda() if USE_CUDA else x


class DQN(Trainer):

    def __init__(self):
        super(DQN, self).__init__()
        self.replay_buffer = ReplayBuffer(self.buffersize)

    def push_to_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def compute_td_loss(self, batch_size, *args):
        state, action, reward, next_state, done = self.replay_buffer.sample(
            batch_size)

        state = Variable(torch.FloatTensor(np.float32(state)))
        next_state = Variable(torch.FloatTensor(np.float32(next_state)))
        action = Variable(torch.LongTensor(action))
        reward = Variable(torch.FloatTensor(reward))
        done = Variable(torch.FloatTensor(done))

        q_values            = self.current_model(state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        next_q_values       = self.current_model(next_state)
        next_q_state_values = self.target_model(next_state)
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = (q_value - Variable(expected_q_value.data)).pow(2)
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
