

def getAllPossibleMoves():
    moves = set()
    rows = ['a','b','c','d','e','f','g','h']
    columns = ['1','2','3','4','5','6','7','8']

    def horizontal(row, column):
        return [row + column + row + col for col in columns if col != column]

    def vertical(row, col):
        return [row + column + r + col for r in rows if r != row]

    def diagonal(row, col):
        m = []
        rOffset = rows.index(row)
        cOffset = columns.index(col)
        for i in range(-8,8):
            if i == 0: 
                continue
            rowIdx = rOffset + i
            if rowIdx in range(8) and cOffset+i in range(8):
                m.append(row + col + rows[rowIdx] + columns[cOffset+i])
            if rowIdx in range(8) and cOffset-i in range(8):
                m.append(row + col + rows[rowIdx] + columns[cOffset-i])
        return m
    
    def horse(row, col):
        m = []
        rOffset = rows.index(row)
        cOffset = columns.index(col)
        idxs = [[rOffset + 2, cOffset - 1],
                [rOffset + 2, cOffset + 1],
                [rOffset + 1, cOffset + 2],
                [rOffset - 1, cOffset + 2],
                [rOffset - 2, cOffset + 1],
                [rOffset - 2, cOffset - 1],
                [rOffset - 1, cOffset - 2],
                [rOffset + 1, cOffset - 2]]

        for idx in idxs:
            if idx[0] in range(0,8) and idx[1] in range(0,8):
                m.append(row + col + rows[idx[0]] + columns[idx[1]])
        
        return m

    for row in rows:
        for column in columns:
            moves.update(horizontal(row, column))       # Horizontal
            moves.update(vertical(row, column))         # Vertical
            moves.update(diagonal(row, column))         # Diagonal
            moves.update(horse(row, column))            # Horse

    return list(moves)



if __name__ == "__main__":

    moves = getAllPossibleMoves()
    print("There are {} possible moves ({} per square).".format(len(moves), round(len(moves) / 64,2)))
