# board.py
# -------------------------
# Handles Tic-Tac-Toe board state and game logic
# -------------------------

import numpy as np

class Board:
    def __init__(self):
        self.reset()

    def reset(self):
        """Initialize an empty board."""
        self.state = np.zeros(9, dtype=int)  # 0 = empty, 1 = agent, -1 = opponent
        self.current_player = 1              # Agent starts by default
        return self.state

    def available_moves(self):
        """Return list of available cell indices."""
        return [i for i, val in enumerate(self.state) if val == 0]

    def make_move(self, move):
        """Apply a move for the current player."""
        if self.state[move] != 0:
            raise ValueError("Invalid move: cell already occupied.")
        self.state[move] = self.current_player
        self.current_player *= -1  # switch turns

    def check_winner(self):
        """Return +1 if agent wins, -1 if opponent wins, 0 if draw, None if ongoing."""
        b = self.state.reshape(3,3)
        lines = np.concatenate([
            b,                      # rows
            b.T,                    # columns
            [np.diag(b)],           # main diagonal
            [np.diag(np.fliplr(b))] # anti-diagonal
        ])
        for line in lines:
            if np.all(line == 1):
                return 1
            elif np.all(line == -1):
                return -1

        if 0 not in self.state:
            return 0  # draw
        return None  # game ongoing

    def render(self):
        """Print board nicely for human view."""
        symbols = {1: 'X', -1: 'O', 0: ' '}
        b = [symbols[x] for x in self.state]
        print(f"{b[0]} | {b[1]} | {b[2]}")
        print("--+---+--")
        print(f"{b[3]} | {b[4]} | {b[5]}")
        print("--+---+--")
        print(f"{b[6]} | {b[7]} | {b[8]}")
        print()
