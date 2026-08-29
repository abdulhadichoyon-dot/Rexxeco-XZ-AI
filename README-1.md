# Self-Learning AI (Python + Java)

A reinforcement-learning agent that learns to solve a maze purely from trial
and reward — no labeled training data, no hardcoded strategy. Two learning
algorithms, in increasing sophistication, plus a real Python↔Java bridge.

## What "self-learning" means here

The agent starts knowing nothing about the environment. It is not told the
correct path. It only receives a reward signal after each move (+50 for
reaching the goal, -10 for hitting an obstacle, -0.1 per step to discourage
wandering) and updates its own value estimates from that signal alone. Over
thousands of trial episodes, an optimal strategy emerges on its own — this is
what makes it "self-learning" in the strict, technical sense, as opposed to
a system trained on human-labeled examples.

## Structure

```
python/
  environment.py        GridWorld — the maze the agent learns to solve
  q_learning_agent.py    Tabular Q-learning (the foundational algorithm)
  train_q_learning.py    Trains + demonstrates the Q-learning agent
  neural_network.py      Feedforward NN implemented from scratch in NumPy
  dqn_agent.py            Deep Q-Network using the from-scratch NN
  train_dqn.py            Trains + demonstrates the DQN agent
  dqn_agent_torch.py      PyTorch version of DQN (see note below)
  agent_server.py         TCP server exposing the trained agent to Java
java/
  AgentClient.java        TCP client + hand-rolled JSON parsing
  Main.java               Drives the trained agent from Java, prints the path
```

## Running it

**Q-learning (tabular):**
```bash
cd python && python3 train_q_learning.py
```
Trains 2000 episodes (~seconds), prints progress every 200 episodes, then
demonstrates the learned policy finding the shortest path to the goal.

**DQN (neural network):**
```bash
cd python && python3 train_dqn.py
```
Same idea, but the agent's "brain" is a neural network instead of a lookup
table, letting it generalize between similar states instead of treating
every grid cell as unrelated. Takes longer (more computation per step) but
reaches the same ~95% success rate.

**Java client (needs a JDK, not just a JRE):**
```bash
# Terminal 1:
cd python && python3 agent_server.py

# Terminal 2:
cd java && javac AgentClient.java Main.java && java Main
```
The Java program connects over a socket, and the actual decision-making —
the trained agent — runs entirely in Python. Java is the client, not a
second implementation of the AI.

## Honesty about what was and wasn't tested

This was built in a sandboxed environment with **no internet access** and
**no JDK installed** (only a JRE). That constrained two things, and I'm
flagging both plainly rather than glossing over them:

1. **`dqn_agent_torch.py` was never run.** PyTorch couldn't be installed
   (no network access), so the from-scratch NumPy version (`dqn_agent.py`)
   is the one that's actually trained and verified — twice, in fact, since
   my first attempt had a real bug (see below). The PyTorch file follows
   the exact same interface and should be a drop-in replacement once you
   have `pip install torch` available, but treat it as unverified until you
   run it.

2. **The Java code was never compiled.** `javac` isn't present in this
   sandbox. I verified the wire protocol by capturing real server output
   and tracing the Java parser's exact logic against it in Python
   character-by-character — every field parsed correctly — but that is a
   logic trace, not a compile-and-run. Compile it yourself first thing;
   if it doesn't build cleanly, tell me the exact error.

Everything else — both training runs, the from-scratch neural network, and
the live TCP server actually driving the trained agent to the goal — was
executed for real and the output above is genuine, not illustrative.

### The DQN bug I found and fixed

My first DQN attempt looked reasonable but was actually broken: success
rate hit 51% around episode 300, then collapsed to 14-22% by the end of
training, and the final demo showed the agent hadn't even moved off the
start tile. Diagnosis: `learning_rate=0.01` was too aggressive and caused a
loss spike (visible in the logs) that overwrote what the network had
already learned, and separately, `epsilon` decayed to its floor too early
for DQN's replay buffer to accumulate the experience diversity it needs
(unlike tabular Q-learning, which doesn't need diverse batches — it updates
one cell at a time). Lowering the learning rate to 0.002 and slowing the
epsilon decay fixed it: 96% success rate, confirmed by rerunning the actual
training script, not just a smaller diagnostic snippet.

## Extending this

- **Bigger/harder maze:** change `GridWorld(size=5)` to a larger value —
  the DQN agent should scale better than tabular Q-learning here, since a
  bigger grid means an exponentially bigger table but the same-size network.
- **Different game:** implement a new environment with the same
  `reset()`/`step()`/`render()` interface as `GridWorld`, and both agents
  work unchanged.
- **Real robotics/game state:** swap `state_to_vector()` for real sensor
  readings or game state — this is exactly the pattern DQN was designed for.
