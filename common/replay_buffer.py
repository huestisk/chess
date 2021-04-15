import numpy as np
import random

from collections import deque

class ReplayBuffer(object):     # Author: https://github.com/higgsfield/RL-Adventure/blob/master/2.double%20dqn.ipynb
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
            
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
    def __len__(self):
        return len(self.buffer)


# class PrioritizedReplayBuffer(ReplayBuffer):    # FIXME
#     def __init__(self, size, alpha):
#         """Create Prioritized Replay buffer.
#         Parameters
#         ----------
#         size: int
#             Max number of transitions to store in the buffer. When the buffer
#             overflows the old memories are dropped.
#         alpha: float
#             how much prioritization is used
#             (0 - no prioritization, 1 - full prioritization)
#         See Also
#         --------
#         ReplayBuffer.__init__
#         """
#         super(PrioritizedReplayBuffer, self).__init__(size)
#         assert alpha > 0
#         self._alpha = alpha

#         it_capacity = 1
#         while it_capacity < size:
#             it_capacity *= 2

#         self._it_sum = SumSegmentTree(it_capacity)
#         self._it_min = MinSegmentTree(it_capacity)
#         self._max_priority = 1.0

#     def push(self, *args, **kwargs):
#         """See ReplayBuffer.store_effect"""
#         idx = self._next_idx
#         super(PrioritizedReplayBuffer, self).push(*args, **kwargs)
#         self._it_sum[idx] = self._max_priority ** self._alpha
#         self._it_min[idx] = self._max_priority ** self._alpha

#     def _sample_proportional(self, batch_size):
#         res = []
#         for _ in range(batch_size):
#             # TODO(szymon): should we ensure no repeats?
#             mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
#             idx = self._it_sum.find_prefixsum_idx(mass)
#             res.append(idx)
#         return res

#     def sample(self, batch_size, beta):
#         """Sample a batch of experiences.
#         compared to ReplayBuffer.sample
#         it also returns importance weights and idxes
#         of sampled experiences.
#         Parameters
#         ----------
#         batch_size: int
#             How many transitions to sample.
#         beta: float
#             To what degree to use importance weights
#             (0 - no corrections, 1 - full correction)
#         Returns
#         -------
#         obs_batch: np.array
#             batch of observations
#         act_batch: np.array
#             batch of actions executed given obs_batch
#         rew_batch: np.array
#             rewards received as results of executing act_batch
#         next_obs_batch: np.array
#             next set of observations seen after executing act_batch
#         done_mask: np.array
#             done_mask[i] = 1 if executing act_batch[i] resulted in
#             the end of an episode and 0 otherwise.
#         weights: np.array
#             Array of shape (batch_size,) and dtype np.float32
#             denoting importance weight of each sampled transition
#         idxes: np.array
#             Array of shape (batch_size,) and dtype np.int32
#             idexes in buffer of sampled experiences
#         """
#         assert beta > 0

#         idxes = self._sample_proportional(batch_size)

#         weights = []
#         p_min = self._it_min.min() / self._it_sum.sum()
#         max_weight = (p_min * len(self._storage)) ** (-beta)

#         for idx in idxes:
#             p_sample = self._it_sum[idx] / self._it_sum.sum()
#             weight = (p_sample * len(self._storage)) ** (-beta)
#             weights.append(weight / max_weight)
#         weights = np.array(weights)
#         encoded_sample = self._encode_sample(idxes)
#         return tuple(list(encoded_sample) + [weights, idxes])

#     def update_priorities(self, idxes, priorities):
#         """Update priorities of sampled transitions.
#         sets priority of transition at index idxes[i] in buffer
#         to priorities[i].
#         Parameters
#         ----------
#         idxes: [int]
#             List of idxes of sampled transitions
#         priorities: [float]
#             List of updated priorities corresponding to
#             transitions at the sampled idxes denoted by
#             variable `idxes`.
#         """
#         assert len(idxes) == len(priorities)
#         for idx, priority in zip(idxes, priorities):
#             assert priority > 0
#             assert 0 <= idx < len(self._storage)
#             self._it_sum[idx] = priority ** self._alpha
#             self._it_min[idx] = priority ** self._alpha

#             self._max_priority = max(self._max_priority, priority)