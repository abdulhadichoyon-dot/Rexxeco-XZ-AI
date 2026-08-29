"""
A minimal feedforward neural network, implemented from scratch in NumPy.

This exists because the sandboxed environment this was built in has no
internet access, so PyTorch could not be installed here. Rather than hand
you an untested torch script, this implements exactly what PyTorch would be
doing under the hood: forward pass, backpropagation, gradient descent.

This is a real, working, trainable neural network — not a simplification.
It's what makes the DQN "deep" instead of "tabular": instead of a lookup
table for every (state, action) pair, the network learns a function that
GENERALIZES — it can estimate the value of states it has never exactly seen
before, based on similar states it has. This is what lets RL scale beyond
tiny grids to large/continuous state spaces (images, sensor readings, etc).

A PyTorch-equivalent version (dqn_agent_torch.py) is included separately for
when you have GPU/internet access — same architecture, same interface, but
using autograd instead of manual backprop.
"""

import numpy as np


class DenseLayer:
    def __init__(self, n_inputs: int, n_outputs: int):
        # He initialization: keeps activations from exploding/vanishing at
        # the start of training, since we're using ReLU.
        self.weights = np.random.randn(n_inputs, n_outputs) * np.sqrt(2.0 / n_inputs)
        self.biases = np.zeros((1, n_outputs))

        # Cached for the backward pass.
        self.inputs = None
        self.output = None

        # Gradients, filled in during backward().
        self.d_weights = None
        self.d_biases = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.output = inputs @ self.weights + self.biases
        return self.output

    def backward(self, d_output: np.ndarray) -> np.ndarray:
        # Gradient of loss w.r.t. this layer's weights/biases (for the update step).
        self.d_weights = self.inputs.T @ d_output
        self.d_biases = np.sum(d_output, axis=0, keepdims=True)
        # Gradient to pass back to the previous layer.
        d_inputs = d_output @ self.weights.T
        return d_inputs


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


class SimpleNeuralNetwork:
    """
    A small MLP: input -> hidden (ReLU) -> hidden (ReLU) -> output (linear).
    Linear output because this is used for Q-value regression, not classification —
    Q-values are unbounded real numbers, not probabilities.
    """

    def __init__(self, n_inputs: int, n_hidden: int, n_outputs: int, learning_rate: float = 0.001):
        self.layer1 = DenseLayer(n_inputs, n_hidden)
        self.layer2 = DenseLayer(n_hidden, n_hidden)
        self.layer3 = DenseLayer(n_hidden, n_outputs)
        self.learning_rate = learning_rate

        # Cached pre-activation values, needed for backprop through ReLU.
        self.z1 = None
        self.z2 = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.z1 = self.layer1.forward(x)
        a1 = relu(self.z1)

        self.z2 = self.layer2.forward(a1)
        a2 = relu(self.z2)

        output = self.layer3.forward(a2)
        return output

    def backward(self, d_loss: np.ndarray):
        d_a2 = self.layer3.backward(d_loss)
        d_z2 = d_a2 * relu_derivative(self.z2)

        d_a1 = self.layer2.backward(d_z2)
        d_z1 = d_a1 * relu_derivative(self.z1)

        self.layer1.backward(d_z1)

    def update_weights(self):
        """Vanilla SGD step. (Adam is used in the DQN agent wrapper for stability —
        this method stays simple since the agent applies its own optimizer state
        on top of these raw gradients.)"""
        for layer in (self.layer1, self.layer2, self.layer3):
            layer.weights -= self.learning_rate * layer.d_weights
            layer.biases -= self.learning_rate * layer.d_biases

    def get_params(self):
        return [
            (self.layer1.weights, self.layer1.biases),
            (self.layer2.weights, self.layer2.biases),
            (self.layer3.weights, self.layer3.biases),
        ]

    def set_params(self, params):
        (self.layer1.weights, self.layer1.biases) = params[0]
        (self.layer2.weights, self.layer2.biases) = params[1]
        (self.layer3.weights, self.layer3.biases) = params[2]

    def copy(self):
        """Used to create the DQN 'target network' — a frozen snapshot that
        stabilizes training (see dqn_agent.py for why this matters)."""
        new_net = SimpleNeuralNetwork(
            self.layer1.weights.shape[0],
            self.layer1.weights.shape[1],
            self.layer3.weights.shape[1],
            self.learning_rate,
        )
        new_net.set_params([(w.copy(), b.copy()) for w, b in self.get_params()])
        return new_net
