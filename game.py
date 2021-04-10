import chess
from ai import AI


class Game():

    def __init__(self, game_mode, autoplay=True):
        self.board = chess.Board()
        self.white_next = True
        self.game_mode = game_mode
        self.autoplay = autoplay
        self.white = AI(self.board) \
            if game_mode in ['auto', 'white-auto'] else None
        self.black = AI(self.board) \
            if game_mode in ['auto', 'black-auto'] else None

    def reset(self):
        self.board.reset()
        self.white_next = True

    def play(self, move_name=None):
        legal = False
        result = None
        if move_name is None:
            move = self.white.choose_move() if self.white_next else self.black.choose_move()
        else:
            move = chess.Move.from_uci(move_name)
        # Play move
        if move in self.board.legal_moves:
            legal = True
            self.board.push(move)
            self.white_next = not self.white_next
        # Check if game is over
        if self.board.is_game_over():
            if self.board.is_checkmate():
                result = 1 if self.white_next else 0    # 0 if white wins, 1 if black wins
            elif self.board.is_insufficient_material() or self.board.is_stalemate():
                result = 2                              # draw
        # Next is AI
        elif self.autoplay and ((self.white_next and self.white is not None) or (not self.white_next and self.black is not None)):
            return self.play()

        return legal, result


if __name__ == '__main__':

    game = Game('manual')
    mate_sequence = ['e2e4', 'e7e5', 'd1h5', 'e8e7', 'h5e5']
    white_sequence = mate_sequence[::2]
    black_sequence = mate_sequence[1::2]

    for move in mate_sequence:
        game.play(move)
