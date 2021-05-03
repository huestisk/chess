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


class Trainer():

    def __init__(self, parameters, simple=True):
        self.env = ChessEnv(parameters["rewards"], simple=simple)     # Opponent does not move
        self.load_model()
        self.optimizer = torch.optim.Adam(self.current_model.parameters())

        self.num_frames = parameters["num_frames"]
        self.buffersize = parameters["buffersize"]
        self.batch_size = parameters["batch_size"]

        self.gamma = parameters["gamma"]

        self.epsilon_start = parameters["epsilon_start"]
        self.epsilon_final = parameters["epsilon_final"]
        self.epsilon_decay = parameters["epsilon_decay"]

    def load_model(self):
        # try:
        #     if USE_CUDA:
        #         self.current_model = torch.load("models/model.pkl")
        #         self.target_model = torch.load("models/model.pkl")
        #     else:
        #         self.current_model = torch.load(
        #             "models/model.pkl", map_location={'cuda:0': 'cpu'})
        #         self.target_model = torch.load(
        #             "models/model.pkl", map_location={'cuda:0': 'cpu'})
        # except:
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

    def epsilon_by_frame(self, frame_idx):
        decay = math.exp(-1. * frame_idx / self.epsilon_decay)
        return self.epsilon_final + (self.epsilon_start - self.epsilon_final) * decay

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
        legalNN = 0
        illegal = 0
        illegalNN = 0

        # Training
        for frame_idx in range(1, self.num_frames + 1):
            epsilon = self.epsilon_by_frame(frame_idx)
            # Select action
            action = self.current_model.act(state, epsilon)
            # if exploring allow only some illegal moves
            if action < 0:
                action = -action if random.random() > 0.8 else self.env.getLegalAction()
                # action = self.env.getLegalAction()
            elif self.env.is_legal_action(action):
                legalNN += 1
            else:
                illegalNN += 1
            # Move action
            next_state, reward, done, info = self.env.step(action)
            # Count illegal moves
            if info.startswith('illegal'):
                illegal += 1
                # if frame_idx < 30000: 
                #     done = True  # FIXME: Can learn this way, but only because states are always "the same"
            self.push_to_buffer(state, action, reward, next_state, done)
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
                state = next_state
            # Train
            loss = self.compute_td_loss(self.batch_size, frame_idx)
            if loss is not None:
                losses.append(loss.data.item())
            # Update Target
            if frame_idx % 1000 == 0:
                self.update_target(self.current_model, self.target_model)
                print("{} frames played: {} wins, {} losses, {} draws, {}% illegal, {} legal NN moves ({}%)".format(
                    frame_idx, wins, defeats, draws, 
                    round(100 * illegal / 1000, 2), legalNN, 
                    round(100 * legalNN / (legalNN + illegalNN), 2)))
                illegal = 0
                legalNN = 0
                illegalNN = 0
            # Save the current model
            if frame_idx % 10000 == 0:
                torch.save(self.current_model, "models/model.pkl")

        torch.save(self.current_model, "models/model.pkl")
        print('Training finished.')

        self.plot(frame_idx, all_rewards, losses)
