# game_manager.py
# =====================================================================
# Handles the gameplay loop between two agents
# =====================================================================

import numpy as np
from agents import QLearningAgent, SARSAAgent


class GameManager:
    def __init__(self, agent_x, agent_o, verbose=False):
        """
        Initialize the game with two agents.
        agent_x: first player (X, +1)
        agent_o: second player (O, -1)
        """
        self.agent_x = agent_x
        self.agent_o = agent_o
        self.verbose = verbose
        self.reset_board()

    def reset_board(self):
        """Resets the Tic-Tac-Toe board and state."""
        self.board = np.zeros(9, dtype=int)
        self.current_player = 1
        return self.board

    def check_winner(self):
        """Check if there is a winner or draw."""
        b = self.board.reshape(3, 3)
        lines = np.concatenate([b, b.T, [b.diagonal()], [np.fliplr(b).diagonal()]])
        for line in lines:
            if np.all(line == 1):
                return 1   # X wins
            if np.all(line == -1):
                return -1  # O wins
        if 0 not in self.board:
            return 0       # Draw
        return None        # Game continues

    def play_game(self, train=True):
        """
        Plays one full game between two agents.
        Returns: +1 if X wins, -1 if O wins, 0 if draw
        """
        self.reset_board()
        state = self.board.copy()
        done = False

        while not done:
            agent = self.agent_x if self.current_player == 1 else self.agent_o
            available_moves = [i for i, v in enumerate(self.board) if v == 0]
            
            # Choose action (with training flag)
            if isinstance(agent, (QLearningAgent, SARSAAgent)):
                action = agent.choose_action(self.board, available_moves, training=train)
            else:
                action = agent.choose_action(self.board, available_moves)

            # Apply move
            self.board[action] = self.current_player

            # Check game result
            winner = self.check_winner()
            done = winner is not None

            # Calculate rewards
            if done:
                if winner == 0:
                    reward_x, reward_o = 0.5, 0.5
                elif winner == 1:
                    reward_x, reward_o = 1, -1
                else:
                    reward_x, reward_o = -1, 1
            else:
                reward_x, reward_o = 0, 0

            next_state = self.board.copy()

            # Update learning agents
            if train:
                if self.current_player == 1:
                    if isinstance(self.agent_x, QLearningAgent):
                        self.agent_x.update(state, action, reward_x, next_state, done)
                    elif isinstance(self.agent_x, SARSAAgent) and not done:
                        next_available = [i for i, v in enumerate(next_state) if v == 0]
                        if next_available:
                            next_action = self.agent_x.choose_action(next_state, next_available, training=True)
                            self.agent_x.update(state, action, reward_x, next_state, next_action, done)
                else:
                    if isinstance(self.agent_o, QLearningAgent):
                        self.agent_o.update(state, action, reward_o, next_state, done)
                    elif isinstance(self.agent_o, SARSAAgent) and not done:
                        next_available = [i for i, v in enumerate(next_state) if v == 0]
                        if next_available:
                            next_action = self.agent_o.choose_action(next_state, next_available, training=True)
                            self.agent_o.update(state, action, reward_o, next_state, next_action, done)

            state = next_state
            self.current_player *= -1

        if self.verbose:
            self.render_board()
            if winner == 0:
                print("Draw!")
            elif winner == 1:
                print("X Wins!")
            else:
                print("O Wins!")

        return winner

    def render_board(self):
        """Print board nicely."""
        symbols = {1: 'X', -1: 'O', 0: ' '}
        b = [symbols[x] for x in self.board]
        print(f"\n{b[0]} | {b[1]} | {b[2]}")
        print("--+---+--")
        print(f"{b[3]} | {b[4]} | {b[5]}")
        print("--+---+--")
        print(f"{b[6]} | {b[7]} | {b[8]}\n")