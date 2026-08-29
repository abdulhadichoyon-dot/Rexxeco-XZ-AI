"""
PyTorch version of the DQN agent — functionally equivalent to dqn_agent.py,
but using torch.nn and autograd instead of hand-rolled NumPy backprop.

*** NOT TESTED in this environment (no internet access to install torch). ***
Verify it yourself before relying on it:
    pip install torch
    python3 train_dqn_torch.py

The interface (choose_action, remember, learn, decay_epsilon) intentionally
matches dqn_agent.py exactly, so train_dqn.py works with either agent by
just changing the import — swap:
    from dqn_agent import DQNAgent
to:
    from dqn_agent_torch import DQNAgent
and nothing else in train_dqn.py needs to change.

Advantages over the from-scratch version once you have torch available:
- GPU acceleration (.to('cuda')) for much larger networks/state spaces
- Adam optimizer (adaptive learning rates per-parameter) instead of plain SGD
- Autograd — no manual backprop math to get wrong
- Easy to extend to convolutional layers for image-based state (e.g. raw
  pixels from a game) instead of hand-crafted feature vectors
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_size: int = 64,
        learning_rate: float = 0.002,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.998,
        buffer_capacity: int = 10000,
        batch_size: int = 16,
        target_update_freq: int = 25,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.q_network = QNetwork(state_dim, n_actions, hidden_size).to(self.device)
        self.target_network = QNetwork(state_dim, n_actions, hidden_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target net is never trained directly.

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.learn_step_counter = 0

    def choose_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.BoolTensor(dones).to(self.device)

        # Q-values for the actions actually taken.
        current_q = self.q_network(states_t).gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            max_next_q = self.target_network(next_states_t).max(dim=1)[0]
            targets = rewards_t + self.gamma * max_next_q * (~dones_t)

        loss = self.loss_fn(current_q, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        torch.save(self.q_network.state_dict(), path)

    def load(self, path: str):
        self.q_network.load_state_dict(torch.load(path, map_location=self.device))
        self.target_network.load_state_dict(self.q_network.state_dict())
