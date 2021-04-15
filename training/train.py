import math
import torch
import numpy as np
import matplotlib.pyplot as plt

# import modules
from training.chessEnv import ChessEnv
from models.models import ChessNN
from common.replay_buffer import ReplayBuffer
from common.wrappers import wrap_pytorch

# Parameters
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 5000

# GPU
USE_CUDA = torch.cuda.is_available()
def Variable(x): return x.cuda() if USE_CUDA else x


# Training functions
def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

def epsilon_by_frame(frame_idx):
    return epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

def compute_td_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state))).flatten(1)
    next_state = Variable(torch.FloatTensor(np.float32(next_state))).flatten(1)
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))

    q_values = current_model(state)
    q_value = q_values.gather(1, action.unsqueeze(0)).squeeze(0)

    next_q_values = current_model(next_state)
    next_q_state_values = target_model(next_state)
    next_q_value = next_q_state_values.gather(
        1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


# Init
env = ChessEnv()

try:
    if USE_CUDA:
        current_model = torch.load("model.pkl")
        target_model = torch.load("model.pkl")
    else:
        current_model = torch.load(
            "model.pkl", map_location={'cuda:0': 'cpu'})
        target_model = torch.load(
            "model.pkl", map_location={'cuda:0': 'cpu'})
except:
    current_model = ChessNN(env.observation_space.shape, env.action_space.n)
    target_model = ChessNN(env.observation_space.shape, env.action_space.n)

    if USE_CUDA:
        current_model = current_model.cuda()
        target_model = target_model.cuda()

optimizer = torch.optim.Adam(current_model.parameters())    # lr=0.0001
replay_buffer = ReplayBuffer(10000)


update_target(current_model, target_model)  # sync nets

# Training
num_frames = 1000000
batch_size = 32
gamma = 0.99

losses = []
all_rewards = []
episode_reward = 0

legal_moves = 0
games_played = 0
finished_games = 0
percent_legal = 0
episode_moves = 0
moves_per_game = []		# change to general info about game

state = env.reset()
for frame_idx in range(1, num_frames + 1):
    epsilon = epsilon_by_frame(frame_idx)
    action = current_model.act(state, epsilon)
    if action < 0:
        action = env.getLegalActions()

    next_state, reward, done, info = env.step(action)
    replay_buffer.push(state, action, reward, next_state, done)

    state = next_state  # check if grad is computed
    episode_reward += reward
    episode_moves += 1

    if info == 'legal':
        legal_moves += 1

    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0
        moves_per_game.append(episode_moves)
        episode_moves = 0
        games_played += 1

    if len(replay_buffer) > batch_size:
        loss = compute_td_loss(batch_size)
        losses.append(loss.data.item())

    if frame_idx % 10000 == 0:
        torch.save(current_model, "model.pkl")

    if frame_idx % 1000 == 0:
        update_target(current_model, target_model)
        frames_per_game = round(frame_idx / games_played,
                                2) if games_played != 0 else np.nan
        print("{} frames: {} games have been played ({} frames per game), {} moves were legal ({}%).".format(
            frame_idx, games_played, frames_per_game, legal_moves, round(
                100 * legal_moves / frame_idx, 2)
        ))

torch.save(current_model, "model.pkl")


def plot(frame_idx, rewards, losses):
    plt.figure(figsize=(20, 5))
    plt.subplot(121)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(122)
    plt.title('loss')
    plt.plot(losses)
    plt.show()


plot(frame_idx, all_rewards, losses)
