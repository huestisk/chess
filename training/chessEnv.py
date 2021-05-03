import pickle
import numpy as np
import random

import gym
from gym import spaces

from chessGame import Game
from common.chessState import ChessState

try:
    ACTIONS = pickle.load(open("actions.pkl", "rb"))
except:
    from common.helper_functions import getAllPossibleMoves
    ACTIONS = np.array(getAllPossibleMoves())
    pickle.dump(ACTIONS, open("actions.pkl", "wb"))

N_DISCRETE_ACTIONS = ACTIONS.size


class ChessEnv(gym.Env):

    def __init__(self, rewards, color=True, simple=False):
        super(ChessEnv, self).__init__()
        self.state = ChessState(color)
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.state.shape,), dtype=np.uint8)
        self.simple = simple
        self.reset(color)
        self.rewards = rewards

    def reset(self, color=True):
        self.state.reset(color)
        if color:   # p1 is white
            self.game = Game('black-auto', autoplay=False)
        else:       # p1 is black
            self.game = Game('white-auto', autoplay=False)
            self.state.update(self.game.board)  # Add inital board to state
            if self.simple:
                self.game.play('0000')
            else:
                self.game.play()    # AI
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
            if self.simple:
                _, result = self.game.play('0000')
            else:
                _, result = self.game.play()
        # Choose Reward
        if result is not None:
            done = True
            # p1 wins
            if (result == 0 and self.state.color) or (result == 1 and not self.state.color):
                info = 'win'
                reward = self.rewards[2]
            # p2 wins
            elif (result == 1 and self.state.color) or (result == 0 and not self.state.color):
                info = 'loss'
                reward = self.rewards[3]
            # draw
            elif result == 2:
                info = 'draw'
                reward = self.rewards[4]
        # legal, non-terminal move
        elif legal:
            reward = self.rewards[0]
        # illegal move
        else:
            reward = self.rewards[1]
            # done = True     # end on illegal move
        # Get new state
        observation = self.getCurrentState()
        return observation, reward, done, info

    def getCurrentState(self):
        self.state.update(self.game.board)
        state = self.state.get()
        # # Reconstruct board
        # board = state[:12*64].reshape((12,64))
        # board = np.sum([(idx+1)*val for idx, val in enumerate(board)], axis=0)
        # board = board.reshape((8,8))[::-1]
        return state

    def is_legal_action(self, action):
        action = ACTIONS[action] if isinstance(action, int) or isinstance(action, np.int64) else action
        return self.game.is_legal_move(action)

    def getLegalAction(self):
        legal_actions = [np.where(ACTIONS == move.uci())[0][0]
                         for move in self.game.board.legal_moves
                         if move.uci() in ACTIONS]
        return random.choice(legal_actions) if len(legal_actions) > 0 else 0

    # Debug
    def visualize(self):
        board = self.game.board
        mat = np.array([board.piece_at(square).symbol() if board.piece_at(
            square) else ' ' for square in range(64)])
        print(mat.reshape(8, 8)[::-1])


if __name__ == "__main__":
    env = ChessEnv(False)
    state = env.reset()
    env.step(8)
