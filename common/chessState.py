import torch
import numpy as np
from collections import deque

TIMESTEPS = 1
MAX_PIECE_INDEX = TIMESTEPS * 12 * 64  # 6 pieces per 2 players, 64 squares
BB_SQUARES = [1 << sq for sq in range(64)]


class ChessState():
    # For T = 8 time-steps (zeros before start t < 1) --> N x N x (MT + L)
    # P1 pieces: one layer (binary features) for each piece type (6)
    # P2 (opponent) pieces: one layer for each piece type

    def __init__(self, color=True):
        self.boards = deque(maxlen=TIMESTEPS)
        self.shape = MAX_PIECE_INDEX
        self.reset(color)

    def reset(self, color=True):
        self.boards.clear()
        self.color = color
        self.castling = None
        self.moveCount = None

    def update(self, board):
        # get indices with pieces
        T = np.zeros((12, 64))
        for square in range(64):
            if board.piece_at(square):
                # pawn starts at 1 (since Null is 0)
                piece = (board.piece_at(square).piece_type - 1 + 6 * (not board.piece_at(square).color))
                T[piece][square] = 1
        # # reconstruct to double check
        # tmp = np.array([(idx+1)*val for idx, val in enumerate(T)]) 
        # tmp_board = np.sum(tmp, axis=0).reshape(8,8)
        # print(tmp_board[::-1])
        self.boards.append(T)
        # get rook positions with castling rights
        self.castling = [idx + MAX_PIECE_INDEX for idx, bb in enumerate(
            BB_SQUARES) if board.castling_rights & bb]
        self.moveCount = board.fullmove_number
        # can claim draw if > 100 no progress (automatic at 150)
        self.moves_without_progress = board.halfmove_clock

    def get(self):
        """ 8 x 6 + 4 --> 52 total layers """
        # 8 timesteps, 6 piece types per player, 64 squares #FIXME: 1 timestep
        # 1 castling (which rooks can still castle)
        # 1 player color (1 if white, 0 if black)
        # 1 total move count
        # 1 moves without progress
        # TODO: add repetions (2): repetition count for that position (3 repitions is an autmatic draw)
        pieces = np.concatenate(self.boards)[::-1]
        pieces = np.concatenate(pieces)
        if len(pieces) == MAX_PIECE_INDEX:
            return pieces
        else:
            return np.concatenate((pieces, np.zeros(MAX_PIECE_INDEX-len(pieces), )))

    def visualize(self, board):
        mat = np.array([board.piece_at(square).symbol() if board.piece_at(
            square) else ' ' for square in range(64)])
        print(mat.reshape(8, 8)[::-1])
