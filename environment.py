"""
GridWorld: a minimal but non-trivial environment for testing self-learning agents.

Layout (5x5 default):
    S . . . .
    . # . # .
    . # . # .
    . . . # .
    . # . . G

S = start, G = goal, # = wall/obstacle (stepping on one ends the episode with a penalty)
The agent starts at S and must learn, purely from trial and reward, to reach G.

This is intentionally framework-free (no OpenAI Gym) so the project has zero
external dependencies beyond numpy — but it follows the same API shape
(reset / step / action_space) that gym-style code expects, so swapping in a
real gym environment later is a one-line change, not a rewrite.
"""

import numpy as np
from typing import Tuple


class GridWorld:
    # Actions: 0=up, 1=down, 2=left, 3=right
    ACTION_DELTAS = {
        0: (-1, 0),
        1: (1, 0),
        2: (0, -1),
        3: (0, 1),
    }
    ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

    def __init__(self, size: int = 5, max_steps: int = 100):
        self.size = size
        self.max_steps = max_steps
        self.action_space_n = 4

        self.start = (0, 0)
        self.goal = (size - 1, size - 1)

        # Fixed obstacle layout for a 5x5 board; scales sparsely for other sizes.
        if size == 5:
            self.walls = {(1, 1), (1, 3), (2, 1), (2, 3), (3, 3), (4, 1)}
        else:
            rng = np.random.default_rng(42)
            n_walls = max(1, size)
            candidates = [
                (r, c)
                for r in range(size)
                for c in range(size)
                if (r, c) not in (self.start, self.goal)
            ]
            idx = rng.choice(len(candidates), size=n_walls, replace=False)
            self.walls = {candidates[i] for i in idx}

        self.agent_pos = self.start
        self.steps_taken = 0

    def reset(self) -> Tuple[int, int]:
        self.agent_pos = self.start
        self.steps_taken = 0
        return self.agent_pos

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, dict]:
        """
        Returns (next_state, reward, done, info) — the standard RL step signature.
        """
        if action not in self.ACTION_DELTAS:
            raise ValueError(f"Invalid action: {action}. Must be 0-3.")

        self.steps_taken += 1
        dr, dc = self.ACTION_DELTAS[action]
        r, c = self.agent_pos
        new_r, new_c = r + dr, c + dc

        # Hitting a boundary wastes the step but doesn't end the episode.
        if not (0 <= new_r < self.size and 0 <= new_c < self.size):
            new_r, new_c = r, c
            reward = -1.0
            done = False
            info = {"event": "wall_boundary"}
        elif (new_r, new_c) in self.walls:
            # Stepping on an obstacle ends the episode with a penalty —
            # this is what forces the agent to learn to route around them.
            new_r, new_c = r, c
            reward = -10.0
            done = True
            info = {"event": "hit_obstacle"}
        elif (new_r, new_c) == self.goal:
            reward = 50.0
            done = True
            info = {"event": "reached_goal"}
        else:
            # Small per-step penalty so the agent learns SHORT paths, not just
            # any path — without this, wandering forever is not discouraged.
            reward = -0.1
            done = False
            info = {"event": "moved"}

        self.agent_pos = (new_r, new_c)

        if self.steps_taken >= self.max_steps and not done:
            done = True
            info = {"event": "timeout"}

        return self.agent_pos, reward, done, info

    def render(self) -> str:
        lines = []
        for r in range(self.size):
            row = []
            for c in range(self.size):
                if (r, c) == self.agent_pos:
                    row.append("A")
                elif (r, c) == self.goal:
                    row.append("G")
                elif (r, c) == self.start:
                    row.append("S")
                elif (r, c) in self.walls:
                    row.append("#")
                else:
                    row.append(".")
            lines.append(" ".join(row))
        return "\n".join(lines)

    def state_to_index(self, state: Tuple[int, int]) -> int:
        """Flatten (row, col) into a single index — needed for tabular Q-learning."""
        r, c = state
        return r * self.size + c

    def state_to_vector(self, state: Tuple[int, int]) -> np.ndarray:
        """
        Normalized (row, col) as floats in [0, 1] — the input format a neural
        network needs (continuous features, not a lookup index). This is also
        what makes DQN generalize: a tabular agent treats state (2,3) and
        state (2,4) as totally unrelated table rows, but a network sees they're
        numerically close and can transfer some of what it learned about one
        to the other. On a 5x5 grid, this doesn't matter much — but it's the
        exact property that lets this same code scale to large or continuous
        state spaces where a table would be impossibly large.
        """
        r, c = state
        return np.array([r / (self.size - 1), c / (self.size - 1)], dtype=float)

    @property
    def n_states(self) -> int:
        return self.size * self.size

    @property
    def state_dim(self) -> int:
        return 2  # (row, col) normalized
