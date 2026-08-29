"""
Tabular Q-Learning agent.

This is "self-learning" in the strict sense: the agent receives no labeled
examples of correct behavior. It only receives (state, action, reward, next_state)
tuples from interacting with the environment, and updates its own value
estimates using the Bellman equation. Over thousands of episodes, optimal
behavior emerges purely from trial, error, and reward — nobody tells it the
answer.

Q-learning update rule (the core of the whole thing):
    Q(s,a) <- Q(s,a) + alpha * [reward + gamma * max(Q(s',a')) - Q(s,a)]

    alpha = learning rate      (how much each new experience overrides old belief)
    gamma = discount factor    (how much future reward matters vs immediate reward)
"""

import numpy as np
import random
from typing import Tuple


class QLearningAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # The Q-table: agent's learned estimate of "how good is action a in state s".
        # Starts at zero — the agent knows NOTHING about the environment initially.
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state_idx: int) -> int:
        """
        Epsilon-greedy policy: explore randomly with probability epsilon,
        otherwise exploit the best known action. This tension between
        exploration and exploitation is the central problem in RL — too
        little exploration and the agent gets stuck on a mediocre strategy
        it found early; too much and it never settles into using what it learned.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self.q_table[state_idx]))

    def learn(
        self,
        state_idx: int,
        action: int,
        reward: float,
        next_state_idx: int,
        done: bool,
    ):
        current_q = self.q_table[state_idx, action]

        if done:
            target = reward  # No future reward if the episode has ended.
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state_idx])

        # Move the current estimate toward the target by a fraction (alpha) —
        # this is the actual "learning" step.
        self.q_table[state_idx, action] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self):
        """Explore less as training progresses and the agent knows more."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        np.save(path, self.q_table)

    def load(self, path: str):
        self.q_table = np.load(path)
