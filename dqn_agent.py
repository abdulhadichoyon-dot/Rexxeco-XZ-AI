"""
Deep Q-Network (DQN) agent.

This is the "advanced" upgrade over tabular Q-learning. Two key ideas make
DQN work where plain Q-learning with a neural net would diverge/fail:

1. Experience Replay: instead of learning from each experience once and
   discarding it, store experiences in a buffer and train on random batches
   sampled from it. This breaks the correlation between consecutive
   experiences (which would otherwise bias the network toward whatever it
   just saw) and lets each experience be learned from multiple times.

2. Target Network: use a SEPARATE, frozen copy of the network to compute the
   "target" Q-values during training, and only periodically sync it with the
   live network. Without this, the network is chasing a target that moves
   every single update (since the same network produces both the prediction
   AND the target it's compared against), which is a well-known source of
   training instability/divergence in DQN.
"""

import numpy as np
import random
from collections import deque
from typing import Tuple
from neural_network import SimpleNeuralNetwork


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=float),
            np.array(actions),
            np.array(rewards, dtype=float),
            np.array(next_states, dtype=float),
            np.array(dones, dtype=bool),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_size: int = 64,
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 100,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.q_network = SimpleNeuralNetwork(state_dim, hidden_size, n_actions, learning_rate)
        self.target_network = self.q_network.copy()

        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.learn_step_counter = 0

    def choose_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        state_batch = state.reshape(1, -1)
        q_values = self.q_network.forward(state_batch)
        return int(np.argmax(q_values[0]))

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return None  # Not enough experience yet to form a batch.

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Current network's Q-value predictions for the states we sampled.
        current_q_values = self.q_network.forward(states)

        # Target network computes what the "correct" Q-values should have
        # been — using the FROZEN target network, not the live one, is what
        # prevents the chasing-a-moving-target instability described above.
        next_q_values = self.target_network.forward(next_states)
        max_next_q = np.max(next_q_values, axis=1)

        targets = current_q_values.copy()
        for i in range(self.batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * max_next_q[i]

        # Mean squared error loss, gradient w.r.t. the network's output.
        loss = np.mean((current_q_values - targets) ** 2)
        d_loss = 2 * (current_q_values - targets) / self.batch_size

        self.q_network.backward(d_loss)
        self.q_network.update_weights()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_network = self.q_network.copy()

        return loss

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
