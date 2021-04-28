import pickle
import numpy as np
import random

import gym
from gym import spaces

from chessGame import Game
from common.chessState import ChessState

# Parameters
HEIGHT = 100
WIDTH = 64

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

    def __init__(self, color=True):
        super(ChessEnv, self).__init__()
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(HEIGHT, WIDTH), dtype=np.uint8)
        self.reset(color)

    def reset(self, color=True):
        self.state = ChessState(color)
        if color:   # p1 is white
            self.game = Game('black-auto', autoplay=False)
        else:       # p1 is black
            self.game = Game('white-auto', autoplay=False)
            self.state.update(self.game.board)  # Add inital board to state
            self.game.play()
        return self.getCurrentState()

    def step(self, action):
        done = False
        # Execute action
        move_name = ACTIONS[action]
        legal, result = self.game.play(move_name)
        info = 'legal' if legal else 'illegal'
        # Opponent plays
        if legal and result is None:
            self.state.update(self.game.board)
            _, result = self.game.play()
        # Choose Reward
        if result is not None:
            done = True
            # p1 wins
            if (result == 0 and self.state.color) or (result == 1 and not self.state.color):
                info = 'win'
                reward = REWARDS[2]
            # black wins
            elif (result == 1 and self.state.color) or (result == 0 and not self.state.color):
                info = 'loss'
                reward = REWARDS[3]
            # draw
            elif result == 2:
                info = 'draw'
                reward = REWARDS[4]
        # legal, non-terminal move
        elif legal:
            reward = REWARDS[0]
        # illegal move
        else:                       # illegal move --> TODO: change reward based on if correct piece was chosen
            reward = REWARDS[1]
            done = True             # FIXME: should not terminate game (long term)
        # Get new state
        observation = self.getCurrentState()
        return observation, reward, done, info       

    def getCurrentState(self):
        self.state.update(self.game.board)
        return self.state.get().to_dense()

    def is_legal_action(self, action):
        action = ACTIONS[action] if isinstance(action, int) else action
        return self.game.is_legal_move(action)

    def getLegalAction(self):
        legal_actions = [np.where(ACTIONS == move.uci())[0][0]
                         for move in self.game.board.legal_moves
                         if move.uci() in ACTIONS]
        return random.choice(legal_actions) if len(legal_actions) > 0 else 0


if __name__ == "__main__":
    env = ChessEnv(False)
    state = env.reset()
    env.step(8)
