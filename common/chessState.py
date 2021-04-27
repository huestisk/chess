import torch
import numpy as np

MAX_PIECE_INDEX = 8 * 12 * 64   # Timestep 8, 6 pieces per player, 64 squares
BB_SQUARES = [1 << sq for sq in range(64)]


class ChessState():
    # For T = 8 time-steps (zeros before start t < 1) --> N x N x (MT + L)
    # P1 pieces: one layer (binary features) for each piece type (6)
    # P2 (opponent) pieces: one layer for each piece type

    def __init__(self, color=True):
        self.pieces = []
        self.color = color
        self.castling = None
        self.moveCount = None

    def update(self, board):
        # get rook positions with castling rights
        self.castling = [idx + MAX_PIECE_INDEX for idx, bb in enumerate(
            BB_SQUARES) if board.castling_rights & bb]
        self.moveCount = board.fullmove_number
        # can claim draw if > 100 no progress (automatic at 150)
        self.moves_without_progress = board.halfmove_clock
        # get indices with pieces
        T = [square + 64 * (board.piece_at(square).piece_type - 1 + 6 * (not board.piece_at(square).color))
             for square in range(64) if board.piece_at(square)]
        self.pieces = T + [x + 64 for x in self.pieces if x < MAX_PIECE_INDEX]

    def get(self):
        """ 8 x 6 + 4 --> 52 total layers """
        # 8 timesteps, 6 piece types per player, 64 squares
        # 1 castling (which rooks can still castle)
        # 1 player color (1 if white, 0 if black)
        # 1 total move count
        # 1 moves without progress
        # TODO: add repetions (2): repetition count for that position (3 repitions is an autmatic draw)
        indices = self.pieces + self.castling + \
            list(range(MAX_PIECE_INDEX + 64, MAX_PIECE_INDEX + 4*64))
        indices = np.array([(int(idx / 64), idx % 64) for idx in indices])
        values = np.concatenate((np.ones(len(self.pieces) + len(self.castling)), self.color * np.ones(
            64), self.moveCount * np.ones(64), self.moves_without_progress * np.ones(64)))
        return torch.sparse_coo_tensor(indices.T, values, (100, 64))
