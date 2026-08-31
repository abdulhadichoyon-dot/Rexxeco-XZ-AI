"""
Core transformer building blocks, implemented from scratch in NumPy.

This is what "modern, like Claude" actually means architecturally: Claude
is a decoder-only transformer that predicts the next token using causal
self-attention over everything before it. This file implements that same
mechanism — self-attention, LayerNorm, residual connections — so it can be
applied to reinforcement learning instead of language (see
decision_transformer_agent.py).

Every piece here operates on a single sequence [seq_len, d_model] (no batch
dimension) to keep the implementation small enough to verify correctly by
hand. The training script loops over a batch and accumulates gradients
instead of vectorizing across a batch axis.
"""

import numpy as np


def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class CausalSelfAttention:
    """
    Multi-head scaled dot-product self-attention with a causal mask —
    position i can only attend to positions <= i. This is what makes the
    model autoregressive: exactly the same constraint that lets GPT/Claude
    predict token t+1 using only tokens 1..t, applied here so an agent's
    action at time t can only depend on its own past, not its future.
    """

    def __init__(self, d_model: int, n_heads: int):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        scale = np.sqrt(2.0 / d_model)
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.W_k = np.random.randn(d_model, d_model) * scale
        self.W_v = np.random.randn(d_model, d_model) * scale
        self.W_o = np.random.randn(d_model, d_model) * scale

        self._cache = {}
        self.d_W_q = self.d_W_k = self.d_W_v = self.d_W_o = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        seq_len, d_model = x.shape
        assert d_model == self.d_model

        Q, K, V = x @ self.W_q, x @ self.W_k, x @ self.W_v

        def split_heads(t):
            return t.reshape(seq_len, self.n_heads, self.d_head).transpose(1, 0, 2)

        Qh, Kh, Vh = split_heads(Q), split_heads(K), split_heads(V)

        mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)  # True = future position, masked out

        scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(self.d_head)
        scores = np.where(mask[None, :, :], -1e9, scores)

        A = softmax(scores, axis=-1)
        head_out = A @ Vh

        concat = head_out.transpose(1, 0, 2).reshape(seq_len, self.d_model)
        output = concat @ self.W_o

        self._cache = dict(x=x, Qh=Qh, Kh=Kh, Vh=Vh, A=A, concat=concat, mask=mask, seq_len=seq_len)
        return output

    def backward(self, d_output: np.ndarray) -> np.ndarray:
        c = self._cache
        seq_len = c["seq_len"]

        d_concat = d_output @ self.W_o.T
        self.d_W_o = c["concat"].T @ d_output

        d_head_out = d_concat.reshape(seq_len, self.n_heads, self.d_head).transpose(1, 0, 2)

        d_A = d_head_out @ c["Vh"].transpose(0, 2, 1)
        d_Vh = c["A"].transpose(0, 2, 1) @ d_head_out

        # Softmax backward (per row, per head): d_scores = A * (d_A - sum(d_A * A))
        A = c["A"]
        sum_term = np.sum(d_A * A, axis=-1, keepdims=True)
        d_scores = A * (d_A - sum_term)
        d_scores = np.where(c["mask"][None, :, :], 0.0, d_scores)  # no gradient through masked positions
        d_scores = d_scores / np.sqrt(self.d_head)

        d_Qh = d_scores @ c["Kh"]
        d_Kh = d_scores.transpose(0, 2, 1) @ c["Qh"]

        def merge_heads(t):
            return t.transpose(1, 0, 2).reshape(seq_len, self.d_model)

        d_Q, d_K, d_V = merge_heads(d_Qh), merge_heads(d_Kh), merge_heads(d_Vh)

        x = c["x"]
        self.d_W_q = x.T @ d_Q
        self.d_W_k = x.T @ d_K
        self.d_W_v = x.T @ d_V

        d_x = d_Q @ self.W_q.T + d_K @ self.W_k.T + d_V @ self.W_v.T
        return d_x

    def params_and_grads(self):
        return [
            (self.W_q, "d_W_q"), (self.W_k, "d_W_k"),
            (self.W_v, "d_W_v"), (self.W_o, "d_W_o"),
        ], [getattr(self, g) for _, g in
            [(self.W_q, "d_W_q"), (self.W_k, "d_W_k"), (self.W_v, "d_W_v"), (self.W_o, "d_W_o")]]


class LayerNorm:
    """
    Normalizes each position's feature vector to zero mean / unit variance,
    then applies a learned scale (gamma) and shift (beta). This is one of
    the unglamorous but load-bearing pieces of every modern transformer —
    without it, stacked attention/FFN blocks are much harder to train
    stably. Pre-norm placement (normalize BEFORE each sublayer, not after)
    is what GPT-2 onward uses, and what this implementation follows.
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps
        self._cache = {}
        self.d_gamma = self.d_beta = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        mu = np.mean(x, axis=-1, keepdims=True)
        var = np.mean((x - mu) ** 2, axis=-1, keepdims=True)
        x_hat = (x - mu) / np.sqrt(var + self.eps)
        out = self.gamma * x_hat + self.beta
        self._cache = dict(x=x, mu=mu, var=var, x_hat=x_hat)
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        c = self._cache
        x, mu, var, x_hat = c["x"], c["mu"], c["var"], c["x_hat"]
        N = x.shape[-1]
        std_inv = 1.0 / np.sqrt(var + self.eps)

        self.d_gamma = np.sum(d_out * x_hat, axis=0)
        self.d_beta = np.sum(d_out, axis=0)

        d_x_hat = d_out * self.gamma
        d_var = np.sum(d_x_hat * (x - mu) * -0.5 * std_inv ** 3, axis=-1, keepdims=True)
        d_mu = np.sum(d_x_hat * -std_inv, axis=-1, keepdims=True) + d_var * np.mean(-2.0 * (x - mu), axis=-1, keepdims=True)
        d_x = d_x_hat * std_inv + d_var * 2.0 * (x - mu) / N + d_mu / N
        return d_x

    def params_and_grads(self):
        return [self.gamma, self.beta], [self.d_gamma, self.d_beta]


def cross_entropy_loss_and_grad(logits: np.ndarray, target_idx: int):
    """
    logits: [n_classes]. target_idx: the correct class index.
    Returns (loss, d_logits) — standard softmax cross-entropy.
    """
    probs = softmax(logits)
    loss = -np.log(probs[target_idx] + 1e-12)
    d_logits = probs.copy()
    d_logits[target_idx] -= 1.0
    return loss, d_logits


class AdamOptimizer:
    """
    Adam maintains a running estimate of each parameter's gradient mean
    (m) and variance (v), and uses them to adapt the effective learning
    rate per-parameter. This converges far more reliably than plain SGD
    for transformer-style architectures, which is why every real
    transformer (including, presumably, Claude) is trained with Adam or a
    close variant rather than vanilla SGD — plain SGD was tried on this
    project's DQN earlier and needed careful hand-tuning of a single
    global learning rate to avoid divergence; Adam removes most of that
    fragility by adapting per-parameter.
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, params: list, grads: list):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            if g is None:
                continue
            key = id(p)
            if key not in self.m:
                self.m[key] = np.zeros_like(p)
                self.v[key] = np.zeros_like(p)
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (g ** 2)
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
