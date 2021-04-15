import random
import pickle
import numpy as np
import torch
import chess

from common.chessState import ChessState

# Parameters
try:
    ACTIONS = pickle.load(open("actions.pkl", "rb"))
except:
    from common.helper_functions import getAllPossibleMoves
    ACTIONS = np.array(getAllPossibleMoves())
    pickle.dump(ACTIONS, open("actions.pkl", "wb"))

MODEL_PATH = "models/model.pkl"
    

class ChessAI():

    def __init__(self, board):
        self.board = board
        self.state = ChessState()

        try:
            if torch.cuda.is_available():
                self.model = torch.load(MODEL_PATH)
            else:
                self.model = torch.load(MODEL_PATH, map_location={'cuda:0': 'cpu'})
        except:
            self.model = None

    def getCurrentState(self):
        self.state.update(self.board)
        return self.state.get()

    def choose_move(self):
        move = None
        if self.model:
            state = self.getCurrentState()
            idx = abs(self.model.act(state, epsilon=1.0))  # on-policy
            move = chess.Move.from_uci(ACTIONS[idx])
        if move not in self.board.legal_moves:
            legal_moves = [move for move in self.board.legal_moves]
            move = random.choice(legal_moves) 
        return move