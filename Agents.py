# agents.py
# -------------------------
# Defines multiple agents for Tic-Tac-Toe:
# - RandomAgent
# - ScriptedAgent
# - QLearningAgent
# - SARSAAgent
# -------------------------

import random
import numpy as np

# ---------------------------------------------------------------------
# 🎲 Random Agent
# ---------------------------------------------------------------------
class RandomAgent:
    """Agent that plays random valid moves."""
    def choose_action(self, state, available_moves):
        return random.choice(available_moves)

# ---------------------------------------------------------------------
# 🧠 Scripted Agent
# ---------------------------------------------------------------------
class ScriptedAgent:
    """
    Agent that follows simple strategy:
      1. Win if possible
      2. Block opponent if they can win next
      3. Take center if available
      4. Else, pick random corner or side
    """
    def choose_action(self, state, available_moves):
        b = np.array(state).reshape(3,3)

        # Helper to test lines for win/block
        def can_win(player):
            for i in range(3):
                # rows
                if np.sum(b[i]) == 2 * player and 0 in b[i]:
                    return (i, np.where(b[i]==0)[0][0])
                # columns
                if np.sum(b[:,i]) == 2 * player and 0 in b[:,i]:
                    return (np.where(b[:,i]==0)[0][0], i)
            # diagonals
            if np.sum(np.diag(b)) == 2 * player and 0 in np.diag(b):
                idx = np.where(np.diag(b)==0)[0][0]
                return (idx, idx)
            if np.sum(np.diag(np.fliplr(b))) == 2 * player and 0 in np.diag(np.fliplr(b)):
                idx = np.where(np.diag(np.fliplr(b))==0)[0][0]
                return (idx, 2-idx)
            return None

        # 1. Try to win
        move = can_win(1)
        if move: return move[0]*3 + move[1]

        # 2. Try to block
        move = can_win(-1)
        if move: return move[0]*3 + move[1]

        # 3. Take center if free
        if 4 in available_moves:
            return 4

        # 4. Else, pick random available move
        return random.choice(available_moves)

# ---------------------------------------------------------------------
# 🧩 Q-Learning Agent
# ---------------------------------------------------------------------
class QLearningAgent:
    """Agent that learns via off-policy Q-Learning."""

    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}     # (state, action) -> value
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_state_key(self, state):
        return tuple(state)

    def choose_action(self, state, available_moves):
        state_key = self.get_state_key(state)
        if random.uniform(0, 1) < self.epsilon:
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

# ---------------------------------------------------------------------
# 🧩 SARSA Agent
# ---------------------------------------------------------------------
class SARSAAgent:
    """Agent that learns via on-policy SARSA."""
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_state_key(self, state):
        return tuple(state)

    def choose_action(self, state, available_moves):
        state_key = self.get_state_key(state)
        if random.uniform(0, 1) < self.epsilon:
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

        # SARSA update rule
        if done:
            target = reward
        else:
            target = reward + self.gamma * next_value

        self.q_table[(state_key, action)] = old_value + self.alpha * (target - old_value)
