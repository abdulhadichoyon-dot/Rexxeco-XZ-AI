"""
Decision Transformer: reinforcement learning reframed as sequence
modeling (Chen et al., 2021 — "Decision Transformer: Reinforcement
Learning via Sequence Modeling").

This is the actual, literal "modern... like Claude model" upgrade, not a
loose analogy. Q-learning and DQN both estimate "how good is this action
from this state" (a value function) and act greedily on that estimate.
This agent does something architecturally different: given a sequence of
(target-return, state, previous-action) tuples — its own recent history —
it predicts the next action directly, using the exact same mechanism
(causal self-attention) that a language model uses to predict the next
token from the text it's already seen. There's no Q-table and no
value-function bootstrapping here; it's next-token prediction, just with
"token" redefined as "action."

Trained OFFLINE via supervised sequence prediction on trajectories another
agent already generated — it never interacts with the live environment
during training, only during evaluation. This mirrors how Claude is
trained on existing text rather than generating and learning from its own
conversations in a loop.
"""

import numpy as np
from transformer_core import CausalSelfAttention, LayerNorm, cross_entropy_loss_and_grad, AdamOptimizer
from neural_network import DenseLayer, relu, relu_derivative


class DecisionTransformerAgent:
    def __init__(
        self,
        state_dim: int = 2,
        n_actions: int = 4,
        d_model: int = 16,
        n_heads: int = 2,
        ffn_hidden: int = 32,
        context_len: int = 6,
        return_scale: float = 50.0,
        learning_rate: float = 0.001,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.d_model = d_model
        self.context_len = context_len
        self.return_scale = return_scale  # normalizes return-to-go into a network-friendly range

        # Embeddings: each modality (return-to-go, state, previous action,
        # position) is projected into the same d_model space and summed —
        # the standard way transformer inputs combine heterogeneous signals.
        self.return_embed = DenseLayer(1, d_model)
        self.state_embed = DenseLayer(state_dim, d_model)
        self.action_embed_table = np.random.randn(n_actions + 1, d_model) * 0.1  # +1 for the START token (no previous action)
        self.pos_embed_table = np.random.randn(context_len, d_model) * 0.1

        # One pre-norm transformer block: LN -> Attention -> residual, LN -> FFN -> residual.
        self.ln1 = LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)
        self.ffn1 = DenseLayer(d_model, ffn_hidden)
        self.ffn2 = DenseLayer(ffn_hidden, d_model)
        self.ln_final = LayerNorm(d_model)

        self.output_head = DenseLayer(d_model, n_actions)

        self.optimizer = AdamOptimizer(lr=learning_rate)

        self._cache = {}
        self.d_action_embed_table = None

    def _embed_sequence(self, return_to_go_seq, state_seq, prev_action_seq):
        """Builds the [seq_len, d_model] input by summing embeddings, as GPT-style
        models sum token + positional embeddings."""
        seq_len = len(return_to_go_seq)
        r_in = np.array(return_to_go_seq, dtype=float).reshape(-1, 1) / self.return_scale
        s_in = np.array(state_seq, dtype=float)

        r_emb = self.return_embed.forward(r_in)
        s_emb = self.state_embed.forward(s_in)
        a_emb = self.action_embed_table[np.array(prev_action_seq) + 1]  # +1 shifts START(-1) to index 0
        p_emb = self.pos_embed_table[:seq_len]

        combined = r_emb + s_emb + a_emb + p_emb
        self._cache["action_indices"] = np.array(prev_action_seq) + 1
        return combined

    def forward(self, return_to_go_seq, state_seq, prev_action_seq):
        """Returns action logits for every position in the sequence, shape [seq_len, n_actions]."""
        x = self._embed_sequence(return_to_go_seq, state_seq, prev_action_seq)

        ln1_out = self.ln1.forward(x)
        attn_out = self.attn.forward(ln1_out)
        x = x + attn_out  # residual

        ln2_out = self.ln2.forward(x)
        ffn_hidden_pre = self.ffn1.forward(ln2_out)
        ffn_hidden_act = relu(ffn_hidden_pre)
        ffn_out = self.ffn2.forward(ffn_hidden_act)
        x = x + ffn_out  # residual
        self._cache["ffn_hidden_pre"] = ffn_hidden_pre

        x = self.ln_final.forward(x)
        logits = self.output_head.forward(x)
        return logits

    def train_step(self, return_to_go_seq, state_seq, prev_action_seq, true_actions):
        """
        One supervised training step: predict the action at every position
        given only the history up to and including that position (enforced
        by the causal mask in attention), compare to the actual action
        taken in the offline trajectory data, backprop, update via Adam.
        """
        logits = self.forward(return_to_go_seq, state_seq, prev_action_seq)
        seq_len = logits.shape[0]

        total_loss = 0.0
        d_logits = np.zeros_like(logits)
        for t in range(seq_len):
            loss_t, d_logits_t = cross_entropy_loss_and_grad(logits[t], true_actions[t])
            total_loss += loss_t
            d_logits[t] = d_logits_t
        total_loss /= seq_len
        d_logits /= seq_len

        d_x = self.output_head.backward(d_logits)
        d_x = self.ln_final.backward(d_x)

        d_ffn_out = d_x
        d_ffn_hidden_act = self.ffn2.backward(d_ffn_out)
        d_ffn_hidden_pre = d_ffn_hidden_act * relu_derivative(self._cache["ffn_hidden_pre"])
        d_ln2_out = self.ffn1.backward(d_ffn_hidden_pre)
        d_ln2_out_from_residual = d_x  # residual: gradient flows both through FFN and directly
        d_x_after_block1 = self.ln2.backward(d_ln2_out) + d_ln2_out_from_residual

        d_attn_out = d_x_after_block1
        d_ln1_out = self.attn.backward(d_attn_out)
        d_ln1_out_from_residual = d_x_after_block1
        d_x_embed = self.ln1.backward(d_ln1_out) + d_ln1_out_from_residual

        # Gradient into the embedding sum splits equally to each summed component.
        d_r_emb = d_x_embed
        d_s_emb = d_x_embed
        d_a_emb = d_x_embed
        d_p_emb = d_x_embed

        self.return_embed.backward(d_r_emb)
        self.state_embed.backward(d_s_emb)

        self.d_action_embed_table = np.zeros_like(self.action_embed_table)
        for t, idx in enumerate(self._cache["action_indices"]):
            self.d_action_embed_table[idx] += d_a_emb[t]

        d_pos_embed = np.zeros_like(self.pos_embed_table)
        d_pos_embed[:seq_len] = d_p_emb

        self._update_all_params(d_pos_embed)
        return total_loss

    def _update_all_params(self, d_pos_embed):
        params, grads = [], []

        params += [self.return_embed.weights, self.return_embed.biases]
        grads += [self.return_embed.d_weights, self.return_embed.d_biases]

        params += [self.state_embed.weights, self.state_embed.biases]
        grads += [self.state_embed.d_weights, self.state_embed.d_biases]

        params += [self.action_embed_table]
        grads += [self.d_action_embed_table]

        params += [self.pos_embed_table]
        grads += [d_pos_embed]

        for module in (self.ln1, self.ln2, self.ln_final):
            p, g = module.params_and_grads()
            params += p
            grads += g

        params += [self.attn.W_q, self.attn.W_k, self.attn.W_v, self.attn.W_o]
        grads += [self.attn.d_W_q, self.attn.d_W_k, self.attn.d_W_v, self.attn.d_W_o]

        params += [self.ffn1.weights, self.ffn1.biases, self.ffn2.weights, self.ffn2.biases]
        grads += [self.ffn1.d_weights, self.ffn1.d_biases, self.ffn2.d_weights, self.ffn2.d_biases]

        params += [self.output_head.weights, self.output_head.biases]
        grads += [self.output_head.d_weights, self.output_head.d_biases]

        self.optimizer.step(params, grads)

    def predict_action(self, return_to_go_seq, state_seq, prev_action_seq):
        """Inference: returns the argmax action at the LAST position of the sequence
        (i.e., 'what should I do right now, given my history so far')."""
        logits = self.forward(return_to_go_seq, state_seq, prev_action_seq)
        return int(np.argmax(logits[-1]))
