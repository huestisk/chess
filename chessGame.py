import chess
from chessAI import ChessAI

class Game():

    def __init__(self, game_mode, autoplay=True):
        self.board = chess.Board()
        self.white_next = True
        self.game_mode = game_mode
        self.autoplay = autoplay
        self.white = ChessAI(self.board) \
            if game_mode in ['auto', 'white-auto'] else None
        self.black = ChessAI(self.board) \
            if game_mode in ['auto', 'black-auto'] else None

    def reset(self):
        self.board.reset()
        self.white_next = True

    def play(self, move=None):
        legal = False
        result = None
        # AI plays move
        if move is None and self.white_next and self.white is not None:
            move = self.white.choose_move()
        elif move is None and not self.white_next and self.black is not None:
            move = self.black.choose_move()
        if self.is_legal_move(move) or move == '0000':  # null move is allowed
            legal = True
            # Convert to chess move
            move = chess.Move.from_uci(move) if isinstance(move, str) else move
            # Play move
            self.board.push(move)
            self.white_next = not self.white_next
        # Check if game is over
        if self.board.is_checkmate():
            result = 1 if self.white_next else 0    # 0 if white wins, 1 if black wins
        elif self.board.is_game_over():             # never claims draw
            result = 2                              # draw
        # Next is AI autoplay
        elif self.autoplay and ((self.white_next and self.white is not None) or (not self.white_next and self.black is not None)):
            return self.play()

        return legal, result

    def is_legal_move(self, move):
        move = chess.Move.from_uci(move) if isinstance(move, str) else move
        return move in self.board.legal_moves


if __name__ == '__main__':

    game = Game('manual')
    mate_sequence = ['e2e4', 'e7e5', 'd1h5', 'e8e7', 'h5e5']
    for move in mate_sequence:
        game.play(move)
