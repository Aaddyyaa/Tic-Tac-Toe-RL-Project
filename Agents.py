# agents.py
# =====================================================================
# All agent implementations for Tic-Tac-Toe
# =====================================================================

import random
import numpy as np


class RandomAgent:
    """Agent that plays random valid moves."""
    def __init__(self, name="Random"):
        self.name = name
    
    def choose_action(self, state, available_moves):
        return random.choice(available_moves)


class ScriptedAgent:
    """
    Agent that follows simple strategy:
      1. Win if possible
      2. Block opponent if they can win next
      3. Take center if available
      4. Else, pick random corner or side
    """
    def __init__(self, name="Scripted"):
        self.name = name
    
    def choose_action(self, state, available_moves):
        b = np.array(state).reshape(3, 3)

        def can_win(player):
            for i in range(3):
                # rows
                if np.sum(b[i]) == 2 * player and 0 in b[i]:
                    return (i, np.where(b[i] == 0)[0][0])
                # columns
                if np.sum(b[:, i]) == 2 * player and 0 in b[:, i]:
                    return (np.where(b[:, i] == 0)[0][0], i)
            # diagonals
            if np.sum(np.diag(b)) == 2 * player and 0 in np.diag(b):
                idx = np.where(np.diag(b) == 0)[0][0]
                return (idx, idx)
            if np.sum(np.diag(np.fliplr(b))) == 2 * player and 0 in np.diag(np.fliplr(b)):
                idx = np.where(np.diag(np.fliplr(b)) == 0)[0][0]
                return (idx, 2 - idx)
            return None

        # 1. Try to win
        move = can_win(1)
        if move:
            return move[0] * 3 + move[1]

        # 2. Try to block
        move = can_win(-1)
        if move:
            return move[0] * 3 + move[1]

        # 3. Take center if free
        if 4 in available_moves:
            return 4

        # 4. Else, pick random available move
        return random.choice(available_moves)


class QLearningAgent:
    """Agent that learns via off-policy Q-Learning."""

    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2, name="Q-Learning"):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.name = name

    def get_state_key(self, state):
        return tuple(state)

    def choose_action(self, state, available_moves, training=True):
        state_key = self.get_state_key(state)
        if training and random.uniform(0, 1) < self.epsilon:
            return random.choice(available_moves)
        q_values = [self.q_table.get((state_key, a), 0) for a in available_moves]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(available_moves, q_values) if q == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        state_key = self.get_state_key(state)
        next_key = self.get_state_key(next_state)
        old_value = self.q_table.get((state_key, action), 0)

        if done:
            target = reward
        else:
            future_qs = [self.q_table.get((next_key, a), 0) for a in range(9)]
            target = reward + self.gamma * max(future_qs)

        self.q_table[(state_key, action)] = old_value + self.alpha * (target - old_value)


class SARSAAgent:
    """Agent that learns via on-policy SARSA."""
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2, name="SARSA"):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.name = name

    def get_state_key(self, state):
        return tuple(state)

    def choose_action(self, state, available_moves, training=True):
        state_key = self.get_state_key(state)
        if training and random.uniform(0, 1) < self.epsilon:
            return random.choice(available_moves)
        q_values = [self.q_table.get((state_key, a), 0) for a in available_moves]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(available_moves, q_values) if q == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, next_action, done):
        state_key = self.get_state_key(state)
        next_key = self.get_state_key(next_state)
        old_value = self.q_table.get((state_key, action), 0)
        next_value = self.q_table.get((next_key, next_action), 0)

        if done:
            target = reward
        else:
            target = reward + self.gamma * next_value

        self.q_table[(state_key, action)] = old_value + self.alpha * (target - old_value)


class HumanAgent:
    """Agent controlled by human input."""
    def __init__(self, name="Human"):
        self.name = name
    
    def choose_action(self, state, available_moves):
        print("\nCurrent board state:")
        self.render_board(state)
        print(f"Available moves: {available_moves}")
        
        while True:
            try:
                move = int(input("Enter your move (0-8): "))
                if move in available_moves:
                    return move
                else:
                    print("Invalid move! Choose from available moves.")
            except ValueError:
                print("Please enter a number between 0 and 8.")
    
    def render_board(self, state):
        """Print board with position numbers."""
        symbols = {1: 'X', -1: 'O', 0: ' '}
        b = [symbols[x] if x != 0 else str(i) for i, x in enumerate(state)]
        print(f"\n{b[0]} | {b[1]} | {b[2]}")
        print("--+---+--")
        print(f"{b[3]} | {b[4]} | {b[5]}")
        print("--+---+--")
        print(f"{b[6]} | {b[7]} | {b[8]}\n")