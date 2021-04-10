import random
import pickle
import numpy as np
import torch

# create image
import cv2
import chess.svg
from chessEnv import HEIGHT, WIDTH, N_CHANNELS
from cairosvg import svg2png

model_path = "model.pkl"

# Parameters
try:
    ACTIONS = pickle.load(open("actions.pkl", "rb"))
except:
    from common.helper_functions import getAllPossibleMoves
    ACTIONS = np.array(getAllPossibleMoves())
    pickle.dump(ACTIONS, open("actions.pkl", "wb"))


class AI():

    def __init__(self, board):
        self.board = board
        if torch.cuda.is_available():
            self.model = torch.load(model_path)
        else:
            self.model = torch.load(model_path, map_location={'cuda:0':'cpu'})

    def choose_move(self):
        # Currently chooses random legal move
        legal_moves = [move for move in self.board.legal_moves]
        state = self.getCurrentState()

        if self.model:
            idx = self.model.act(state, epsilon=0)  # on-policy
            move = chess.Move.from_uci(ACTIONS[idx])
            if move not in legal_moves: print("AI choose an illegal move.")
            return move if move in legal_moves else random.choice(legal_moves)
        else:
            return random.choice(legal_moves)

    def getCurrentState(self):
        # TODO: directly create image / current state
        chessboard = chess.svg.board(
            self.board, size=WIDTH, coordinates=False).encode("UTF-8")

        chessboard = svg2png(chessboard)
        chessboard = np.frombuffer(chessboard, np.uint8)
        chessboard = cv2.imdecode(chessboard, cv2.IMREAD_GRAYSCALE)
        chessboard = chessboard.reshape(HEIGHT, WIDTH, N_CHANNELS)

        return np.swapaxes(chessboard, 2, 0)
