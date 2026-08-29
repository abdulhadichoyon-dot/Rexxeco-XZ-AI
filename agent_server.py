"""
TCP server exposing a trained agent's decision-making over a socket.

This is the actual Python<->Java bridge: Java has no serious RL/ML
ecosystem, so the pattern used in real systems is exactly this — Python
owns training and inference, and other languages (Java, in a game client,
Android app, backend service, etc) talk to it over a simple protocol.

Protocol: newline-delimited JSON, one request/response per line.
    Request:  {"row": 2, "col": 3}
    Response: {"action": 3, "action_name": "RIGHT", "row": 2, "col": 4, "reward": -0.1, "done": false}

The server holds ONE live GridWorld environment and trained Q-learning
agent. Each request steps that environment forward.

Run: python3 agent_server.py [port]
"""

import socket
import json
import sys
import numpy as np
from environment import GridWorld
from q_learning_agent import QLearningAgent


def load_or_train_agent(env: GridWorld) -> QLearningAgent:
    agent = QLearningAgent(n_states=env.n_states, n_actions=env.action_space_n)
    try:
        agent.load("/home/claude/self-learning-ai/python/q_table_trained.npy")
        agent.epsilon = 0.0  # Pure exploitation — this is a trained agent being served, not one still learning.
        print("Loaded pre-trained Q-table.")
    except FileNotFoundError:
        print("No trained Q-table found — training a fresh agent now (this takes a few seconds)...")
        for episode in range(2000):
            state = env.reset()
            state_idx = env.state_to_index(state)
            done = False
            while not done:
                action = agent.choose_action(state_idx)
                next_state, reward, done, info = env.step(action)
                next_state_idx = env.state_to_index(next_state)
                agent.learn(state_idx, action, reward, next_state_idx, done)
                state_idx = next_state_idx
            agent.decay_epsilon()
        agent.epsilon = 0.0
        agent.save("/home/claude/self-learning-ai/python/q_table_trained.npy")
        print("Training complete.")
    return agent


def handle_client(conn: socket.socket, env: GridWorld, agent: QLearningAgent):
    buffer = ""
    with conn:
        while True:
            data = conn.recv(4096)
            if not data:
                break
            buffer += data.decode("utf-8")

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue

                try:
                    request = json.loads(line)
                except json.JSONDecodeError:
                    error_resp = json.dumps({"error": "invalid JSON"}) + "\n"
                    conn.sendall(error_resp.encode("utf-8"))
                    continue

                if "reset" in request and request["reset"]:
                    state = env.reset()
                    response = {
                        "row": state[0],
                        "col": state[1],
                        "reward": 0.0,
                        "done": False,
                        "event": "reset",
                    }
                else:
                    row = request.get("row")
                    col = request.get("col")
                    if row is None or col is None:
                        response = {"error": "request must include row and col, or reset:true"}
                    else:
                        # The server treats the incoming (row, col) as ground
                        # truth for where the calling client's game/agent
                        # currently is, looks up what the trained agent would
                        # do from there, and returns the resulting transition.
                        env.agent_pos = (row, col)
                        state_idx = env.state_to_index((row, col))
                        action = agent.choose_action(state_idx)
                        next_state, reward, done, info = env.step(action)

                        response = {
                            "action": action,
                            "action_name": env.ACTION_NAMES[action],
                            "row": next_state[0],
                            "col": next_state[1],
                            "reward": reward,
                            "done": done,
                            "event": info.get("event"),
                        }

                conn.sendall((json.dumps(response) + "\n").encode("utf-8"))


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5555

    env = GridWorld(size=5, max_steps=100)
    agent = load_or_train_agent(env)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", port))
    server.listen(5)
    print(f"Agent server listening on port {port}...")
    print("Waiting for Java client connection.\n")

    try:
        while True:
            conn, addr = server.accept()
            print(f"Client connected: {addr}")
            handle_client(conn, env, agent)
            print(f"Client disconnected: {addr}")
    except KeyboardInterrupt:
        print("\nShutting down server.")
    finally:
        server.close()


if __name__ == "__main__":
    main()
