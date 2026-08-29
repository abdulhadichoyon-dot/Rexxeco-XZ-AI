"""
Trains the DQN agent on GridWorld using continuous state vectors instead of
a lookup table. Structured to mirror train_q_learning.py so the two
approaches are directly comparable.

Run: python3 train_dqn.py
"""

import numpy as np
from environment import GridWorld
from dqn_agent import DQNAgent


def train(n_episodes: int = 2000, verbose: bool = True):
    env = GridWorld(size=5, max_steps=100)
    # These hyperparameters were tuned empirically (see dev notes): the
    # original lr=0.01 caused loss spikes and policy collapse after ~300
    # episodes, and epsilon decaying to its floor by episode ~600 starved
    # the replay buffer of diverse experience before the network had
    # useful signal to learn from. lr=0.002 + slower epsilon_decay=0.998
    # over more episodes converges smoothly to ~95% success, matching the
    # tabular agent's ceiling on this problem size.
    agent = DQNAgent(
        state_dim=env.state_dim,
        n_actions=env.action_space_n,
        hidden_size=32,
        learning_rate=0.002,
        target_update_freq=25,
        batch_size=16,
        epsilon_decay=0.998,
    )

    episode_rewards = []
    episode_lengths = []
    losses = []
    success_count = 0

    for episode in range(n_episodes):
        state = env.reset()
        state_vec = env.state_to_vector(state)
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action = agent.choose_action(state_vec)
            next_state, reward, done, info = env.step(action)
            next_state_vec = env.state_to_vector(next_state)

            agent.remember(state_vec, action, reward, next_state_vec, done)
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)

            state_vec = next_state_vec
            total_reward += reward
            steps += 1

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        if info.get("event") == "reached_goal":
            success_count += 1

        if verbose and (episode + 1) % 250 == 0:
            window = min(250, episode + 1)
            recent_successes = sum(
                1 for i in range(max(0, episode - window + 1), episode + 1) if episode_rewards[i] > 0
            )
            avg_reward = np.mean(episode_rewards[-window:])
            avg_len = np.mean(episode_lengths[-window:])
            avg_loss = np.mean(losses[-500:]) if losses else 0.0
            print(
                f"Episode {episode + 1:4d} | "
                f"Avg reward (last {window}): {avg_reward:6.2f} | "
                f"Avg steps: {avg_len:5.1f} | "
                f"Success rate: {recent_successes / window * 100:5.1f}% | "
                f"Avg loss: {avg_loss:.3f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    return agent, env, episode_rewards, episode_lengths, success_count


def demonstrate_learned_policy(agent: DQNAgent, env: GridWorld):
    print("\n" + "=" * 50)
    print("DEMONSTRATION: DQN agent following its learned policy")
    print("=" * 50)

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    state = env.reset()
    state_vec = env.state_to_vector(state)
    done = False
    path = [state]

    print(f"\nStart:\n{env.render()}\n")

    step_num = 0
    while not done and step_num < 30:
        action = agent.choose_action(state_vec)
        next_state, reward, done, info = env.step(action)
        state_vec = env.state_to_vector(next_state)
        path.append(next_state)
        step_num += 1

        if info.get("event") == "reached_goal":
            print(f"Reached goal in {step_num} steps! Path: {path}")
        elif done:
            print(f"Episode ended ({info.get('event')}) after {step_num} steps. Path: {path}")

    print(f"\nFinal state:\n{env.render()}")
    agent.epsilon = old_epsilon


if __name__ == "__main__":
    print("Training DQN agent on GridWorld (neural network, from scratch)...\n")
    agent, env, rewards, lengths, successes = train(n_episodes=2000)

    print(f"\nTraining complete. Reached goal in {successes}/2000 episodes.")
    print(f"Final 100-episode average reward: {np.mean(rewards[-100:]):.2f}")

    demonstrate_learned_policy(agent, env)
