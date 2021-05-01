import math
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

# import modules
from training.chessEnv import ChessEnv
from models.models import ChessNN

# Use GPU, if available
USE_CUDA = torch.cuda.is_available()

# Training parameters
num_frames = 1000000
batch_size = 32
gamma = 0.99

buffersize = 50000

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500000

beta_start = 0.4
beta_frames = 100000
alpha = 0.6

# Rewards:     move,   illegal,    win,    loss,   draw
rewards = [    1e-4,    0,          1,      -1,     0   ]

def epsilon_by_frame(frame_idx):
    decay = math.exp(-1. * frame_idx / epsilon_decay)
    return epsilon_final + (epsilon_start - epsilon_final) * decay


def beta_by_frame(frame_idx):
    beta = beta_start + frame_idx * (1.0 - beta_start) / beta_frames
    return min(1.0, beta)


class Trainer():

    def __init__(self):
        self.env = ChessEnv(rewards, simple=True)     # Opponent does not move
        self.load_model()
        self.optimizer = torch.optim.Adam(self.current_model.parameters())
        self.buffersize = buffersize
        self.alpha = alpha
        self.gamma = gamma

    def load_model(self):
        try:
            if USE_CUDA:
                self.current_model = torch.load("models/model.pkl")
                self.target_model = torch.load("models/model.pkl")
            else:
                self.current_model = torch.load(
                    "models/model.pkl", map_location={'cuda:0': 'cpu'})
                self.target_model = torch.load(
                    "models/model.pkl", map_location={'cuda:0': 'cpu'})
        except:
            self.current_model = ChessNN(
                self.env.observation_space.shape, self.env.action_space.n)
            self.target_model = ChessNN(
                self.env.observation_space.shape, self.env.action_space.n)

            if USE_CUDA:
                self.current_model = self.current_model.cuda()
                self.target_model = self.target_model.cuda()

        self.update_target(self.current_model, self.target_model)  # sync nets

    def update_target(self, current_model, target_model):
        self.target_model.load_state_dict(self.current_model.state_dict())

    def plot(self, frame_idx, rewards, losses):
        plt.figure(figsize=(20, 5))
        plt.subplot(121)
        plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
        plt.plot(rewards)
        plt.subplot(122)
        plt.title('loss')
        plt.plot(losses)
        plt.show()

    def push_to_buffer(self, *args):
        pass

    def compute_td_loss(self, *args):
        return None

    def train(self):
        # Variables
        color = True                    # random.random() > 0.5  # white if True
        state = self.env.reset(color)
        losses = []
        all_rewards = []
        episode_reward = 0

        # Log
        wins = 0
        defeats = 0
        draws = 0
        terminations = 0
        legal = 0
        illegal = 0

        # Training
        for frame_idx in range(1, num_frames + 1):
            epsilon = epsilon_by_frame(frame_idx)
            # Select action
            action = self.current_model.act(state, epsilon)
            # if exploring allow only some illegal moves
            if action < 0:
                action = -action if random.random() > 0.5 else self.env.getLegalAction()
                # action = self.env.getLegalAction()
            elif self.env.is_legal_action(action):
                legal += 1
            # Move action
            next_state, reward, done, info = self.env.step(action)
            self.push_to_buffer(state, action, reward, next_state, done)
            # Count illegal moves
            if info.startswith('illegal'):
                illegal += 1
                done =  True    #FIXME
            # Accumulate rewards
            episode_reward += reward
            # Check if game has been terminated
            if done:
                color = True  # random.random() > 0.5  # randomly choose player color
                state = self.env.reset(color)
                all_rewards.append(episode_reward)
                episode_reward = 0
                if info == 'win':
                    wins += 1
                elif info == 'loss':
                    defeats += 1
                elif info == 'draw':
                    draws += 1
                else:
                    terminations += 1
            else:
                state = next_state
            # Train
            beta = beta_by_frame(frame_idx)
            loss = self.compute_td_loss(batch_size, beta)
            if loss is not None:
                losses.append(loss.data.item())
            # Save the current model
            if frame_idx % 10000 == 0:
                torch.save(self.current_model, "models/model.pkl")
            # Update Target
            if frame_idx % 1000 == 0:
                self.update_target(self.current_model, self.target_model)
                print("{} frames played: {} wins, {} losses, {} draws, {} terminations, {} legal NN moves, {} illegal moves.".format(
                    frame_idx, wins, defeats, draws, terminations, legal, illegal))

        torch.save(self.current_model, "models/model.pkl")
        print('Training finished.')

        self.plot(frame_idx, all_rewards, losses)
