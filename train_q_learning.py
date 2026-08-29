"""
Trains the tabular Q-learning agent on GridWorld and reports learning progress.

Run: python3 train_q_learning.py
"""

import numpy as np
from environment import GridWorld
from q_learning_agent import QLearningAgent


def train(n_episodes: int = 2000, verbose: bool = True):
    env = GridWorld(size=5, max_steps=100)
    agent = QLearningAgent(n_states=env.n_states, n_actions=env.action_space_n)

    episode_rewards = []
    episode_lengths = []
    success_count = 0

    for episode in range(n_episodes):
        state = env.reset()
        state_idx = env.state_to_index(state)
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action = agent.choose_action(state_idx)
            next_state, reward, done, info = env.step(action)
            next_state_idx = env.state_to_index(next_state)

            agent.learn(state_idx, action, reward, next_state_idx, done)

            state_idx = next_state_idx
            total_reward += reward
            steps += 1

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        if info.get("event") == "reached_goal":
            success_count += 1

        if verbose and (episode + 1) % 200 == 0:
            recent_success_rate = (
                sum(
                    1
                    for i in range(max(0, episode - 199), episode + 1)
                    if episode_rewards[i] > 0
                )
                / min(200, episode + 1)
                * 100
            )
            avg_reward = np.mean(episode_rewards[-200:])
            avg_len = np.mean(episode_lengths[-200:])
            print(
                f"Episode {episode + 1:4d} | "
                f"Avg reward (last 200): {avg_reward:6.2f} | "
                f"Avg steps: {avg_len:5.1f} | "
                f"Success rate: {recent_success_rate:5.1f}% | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    return agent, env, episode_rewards, episode_lengths, success_count


def demonstrate_learned_policy(agent: QLearningAgent, env: GridWorld):
    """Run one episode with epsilon=0 (pure exploitation) to show what was learned."""
    print("\n" + "=" * 50)
    print("DEMONSTRATION: Agent following its learned policy")
    print("=" * 50)

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # No randomness — show exactly what it learned.

    state = env.reset()
    state_idx = env.state_to_index(state)
    done = False
    path = [state]

    print(f"\nStart:\n{env.render()}\n")

    step_num = 0
    while not done and step_num < 30:
        action = agent.choose_action(state_idx)
        next_state, reward, done, info = env.step(action)
        state_idx = env.state_to_index(next_state)
        path.append(next_state)
        step_num += 1

        if info.get("event") == "reached_goal":
            print(f"Reached goal in {step_num} steps! Path: {path}")
        elif done:
            print(f"Episode ended ({info.get('event')}) after {step_num} steps. Path: {path}")

    print(f"\nFinal state:\n{env.render()}")
    agent.epsilon = old_epsilon


if __name__ == "__main__":
    print("Training Q-Learning agent on GridWorld...\n")
    agent, env, rewards, lengths, successes = train(n_episodes=2000)

    print(f"\nTraining complete. Reached goal in {successes}/2000 episodes.")
    print(f"Final 100-episode average reward: {np.mean(rewards[-100:]):.2f}")

    demonstrate_learned_policy(agent, env)

    agent.save("/home/claude/self-learning-ai/python/q_table_trained.npy")
    print("\nSaved trained Q-table to q_table_trained.npy")
