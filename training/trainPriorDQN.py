import torch
import numpy as np
from training.training import Trainer
from common.replay_buffer import PrioritizedReplayBuffer


# Use GPU, if available
USE_CUDA = torch.cuda.is_available()
def Variable(x): return x.cuda() if USE_CUDA else x


class PriorDQN(Trainer):

    def __init__(self, alpha=0.6):
        super(PriorDQN, self).__init__()
        self.replay_buffer = PrioritizedReplayBuffer(self.buffersize, self.alpha)

    def push_to_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def compute_td_loss(self, batch_size, beta):

        if len(self.replay_buffer) < 1000:
            return None
        
        state, action, reward, next_state, done, indices, weights = self.replay_buffer.sample(
            batch_size, beta)

        state = Variable(torch.FloatTensor(np.float32(state))).flatten(1)
        next_state = Variable(torch.FloatTensor(
            np.float32(next_state))).flatten(1)
        action = Variable(torch.LongTensor(action))
        reward = Variable(torch.FloatTensor(reward))
        done = Variable(torch.FloatTensor(done))
        weights = Variable(torch.FloatTensor(weights))

        q_values = self.current_model(state)
        q_value = q_values.gather(1, action.unsqueeze(0)).squeeze(0)

        next_q_values = self.current_model(next_state)
        next_q_state_values = self.target_model(next_state)
        next_q_value = next_q_state_values.gather(
            1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = (q_value - Variable(expected_q_value.data)).pow(2) * weights
        prios = loss + 1e-5
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
        self.optimizer.step()

        return loss
