
import torch
import numpy as np


class ChessState():
    # For T = 8 time-steps (zeros before start t < 1) --> N x N x (MT + L)
    # P1 pieces: one layer (binary features) for each piece type (6)
    # P2 (opponent) pieces: one layer for each piece type

    # TODO:
    # Repetions (2): repetition count for that position (3 repitions is an autmatic draw)
    # players color (1)
    # total move count (1)
    # legality of castling (kingside, queenside), P1 (2), P2 (2)
    # number of moves without progress (50 moves automatic draw), (1)

    def __init__(self):
        self.color = True  # TODO: allow black AI
        self.indices = None
        self.values = None  # TODO: add non-binary layers

    def update(self, board):
        T = np.array([np.array([0, 6 * (not board.piece_at(square).color) + board.piece_at(
            square).piece_type - 1, square]) for square in range(64) if board.piece_at(square)])
        
        if self.indices is not None:
            self.indices = self.indices[self.indices[:,0] < 6]
            self.indices[:,0] += 1
            self.indices = np.concatenate((self.indices, T), axis=0)
        else:
            self.indices = T

    def get(self):
        # 8 timesteps, 6 piece types per player, 64 squares
        v = np.ones(len(self.indices))
        return torch.sparse_coo_tensor(self.indices.T, v, (8, 12, 64))
