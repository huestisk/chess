import pickle
import numpy as np
import random

import gym
from gym import spaces

from chessGame import Game
from common.chessState import ChessState

# Parameters
HEIGHT = 12
WIDTH = 64
N_CHANNELS = 8

try:
    ACTIONS = pickle.load(open("actions.pkl", "rb"))
except:
    from common.helper_functions import getAllPossibleMoves
    ACTIONS = np.array(getAllPossibleMoves())
    pickle.dump(ACTIONS, open("actions.pkl", "wb"))

N_DISCRETE_ACTIONS = ACTIONS.size

# FIXME: Move reward should be changed to negative, once AI learns to do legal moves
REWARDS = [5e-3, -1, 1, -1, 0]      # move, illegal move, win, loss, draw


class ChessEnv(gym.Env):

    def __init__(self):
        super(ChessEnv, self).__init__()
        self.game = Game('black-auto')      # TODO: add different players
        self.state = ChessState()
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
        
    def step(self, action):
        done = False
        # Execute action
        move_name = ACTIONS[action]
        legal, result = self.game.play(move_name)
        info = 'legal' if legal else 'illegal'
        # Choose Reward
        if result is not None:
            done = True
            if result == 0:         # white wins --> TODO: match player color
                info += ', white wins'
                reward = REWARDS[2]
            elif result == 1:       # black wins
                info += ', black wins'
                reward = REWARDS[3]
            elif result == 2:       # draw
                info += ', draw'
                reward = REWARDS[4]
        elif legal:
            reward = REWARDS[0]
        else:           # illegal move --> TODO: change reward based on if correct piece was chosen
            done = True  # FIXME
            reward = REWARDS[1]

        # Get new state
        observation = self.getCurrentState()     # Next state

        return observation, reward, done, info

    def reset(self):
        self.game.reset()
        return self.getCurrentState()

    def getCurrentState(self):
        self.state.update(self.game.board)
        return self.state.get().to_dense()

    def getLegalActions(self):
        # FIXME: some legal actions are not in ACTIONS (e.g. 'g7h8q')
        legal_actions = [np.where(ACTIONS == move.uci())[0][0] 
            for move in self.game.board.legal_moves 
            if move.uci() in ACTIONS]  
        return random.choice(legal_actions)



if __name__ == "__main__":
    env = ChessEnv()
    state = env.reset()
    env.step(8)


