"""Qwen3.5 text forward pass in NumPy float32.

A transcription of transformers' modeling_qwen3_5.py (text model only). One
code path serves prefill and decode: `forward` appends tokens to a `State`,
running the GatedDeltaNet recurrence token by token.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .safetensors import Checkpoint

PREFIX = "model.language_model."


@dataclass(frozen=True)
class Config:
    hidden: int
    intermediate: int
    vocab: int
    eps: float
    layer_types: tuple[str, ...]
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rotary_dim: int
    rope_theta: float
    lin_k_heads: int
    lin_v_heads: int
    lin_k_dim: int
    lin_v_dim: int
    conv_kernel: int
    eos_token_id: int

    @classmethod
    def load(cls, directory: Path) -> "Config":
        t = json.loads((directory / "config.json").read_text())["text_config"]
        rope = t["rope_parameters"]
        return cls(
            hidden=t["hidden_size"],
            intermediate=t["intermediate_size"],
            vocab=t["vocab_size"],
            eps=t["rms_norm_eps"],
            layer_types=tuple(t["layer_types"]),
            num_heads=t["num_attention_heads"],
            num_kv_heads=t["num_key_value_heads"],
            head_dim=t["head_dim"],
            rotary_dim=int(t["head_dim"] * rope.get("partial_rotary_factor", 1.0)),
            rope_theta=float(rope["rope_theta"]),
            lin_k_heads=t["linear_num_key_heads"],
            lin_v_heads=t["linear_num_value_heads"],
            lin_k_dim=t["linear_key_head_dim"],
            lin_v_dim=t["linear_value_head_dim"],
            conv_kernel=t["linear_conv_kernel_dim"],
            eos_token_id=t["eos_token_id"],
        )


@dataclass
class LayerState:
    k: np.ndarray | None = None  # full attention: [T, kv_heads, head_dim], after QK-norm and RoPE
    v: np.ndarray | None = None
    conv: np.ndarray | None = None  # GatedDeltaNet: last (kernel - 1) pre-conv qkv rows
    s: np.ndarray | None = None  # GatedDeltaNet: [heads, k_dim, v_dim]


@dataclass
class State:
    pos: int = 0
    layers: list[LayerState] = field(default_factory=list)


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def rms_norm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    """Zero-centred RMSNorm: x̂ · (1 + w)."""
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * (1.0 + w)


def l2_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + eps)


def rotate_half(x: np.ndarray) -> np.ndarray:
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)


class Qwen35:
    def __init__(self, directory: Path):
        self.cfg = Config.load(directory)
        ckpt = Checkpoint(directory)
        g = lambda name: ckpt.get(PREFIX + name).astype(np.float32, copy=False)  # noqa: E731
        c = self.cfg
        self.embed = g("embed_tokens.weight")
        self.final_norm = g("norm.weight")
        self.layers = []
        for i, kind in enumerate(c.layer_types):
            p = f"layers.{i}."
            w = {
                "in_norm": g(p + "input_layernorm.weight"),
                "post_norm": g(p + "post_attention_layernorm.weight"),
                "gate": g(p + "mlp.gate_proj.weight"),
                "up": g(p + "mlp.up_proj.weight"),
                "down": g(p + "mlp.down_proj.weight"),
            }
            if kind == "full_attention":
                a = p + "self_attn."
                for n in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
                    w[n] = g(a + n + ".weight")
            else:
                a = p + "linear_attn."
                for n in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj", "norm"):
                    w[n] = g(a + n + ".weight")
                w["conv"] = g(a + "conv1d.weight")[:, 0, :]  # [channels, kernel]
                w["dt_bias"] = g(a + "dt_bias")
                w["a_log"] = g(a + "A_log")
            self.layers.append(w)
        half = c.rotary_dim // 2
        self.inv_freq = 1.0 / (c.rope_theta ** (np.arange(0, c.rotary_dim, 2, dtype=np.float64) / c.rotary_dim))
        assert self.inv_freq.shape == (half,)

    def new_state(self) -> State:
        c = self.cfg
        conv_dim = 2 * c.lin_k_heads * c.lin_k_dim + c.lin_v_heads * c.lin_v_dim
        layers = []
        for kind in c.layer_types:
            if kind == "full_attention":
                empty = np.zeros((0, c.num_kv_heads, c.head_dim), np.float32)
                layers.append(LayerState(k=empty, v=empty.copy()))
            else:
                layers.append(
                    LayerState(
                        conv=np.zeros((c.conv_kernel - 1, conv_dim), np.float32),
                        s=np.zeros((c.lin_v_heads, c.lin_k_dim, c.lin_v_dim), np.float32),
                    )
                )
        return State(pos=0, layers=layers)

    # -- mixers ---------------------------------------------------------------

    def _rope(self, x: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """x: [T, heads, head_dim]; rotates the first rotary_dim dims (rotate-half layout)."""
        r = self.cfg.rotary_dim
        freqs = positions[:, None].astype(np.float64) * self.inv_freq[None, :]
        emb = np.concatenate([freqs, freqs], axis=-1)
        cos = np.cos(emb).astype(np.float32)[:, None, :]
        sin = np.sin(emb).astype(np.float32)[:, None, :]
        rot, rest = x[..., :r], x[..., r:]
        return np.concatenate([rot * cos + rotate_half(rot) * sin, rest], axis=-1)

    def _attention(self, w: dict, x: np.ndarray, st: LayerState, pos0: int) -> np.ndarray:
        c = self.cfg
        T = x.shape[0]
        qg = (x @ w["q_proj"].T).reshape(T, c.num_heads, 2 * c.head_dim)
        q, gate = qg[..., : c.head_dim], qg[..., c.head_dim :].reshape(T, -1)
        k = (x @ w["k_proj"].T).reshape(T, c.num_kv_heads, c.head_dim)
        v = (x @ w["v_proj"].T).reshape(T, c.num_kv_heads, c.head_dim)
        q = rms_norm(q, w["q_norm"], c.eps)
        k = rms_norm(k, w["k_norm"], c.eps)
        positions = np.arange(pos0, pos0 + T)
        q = self._rope(q, positions)
        k = self._rope(k, positions)
        st.k = np.concatenate([st.k, k])
        st.v = np.concatenate([st.v, v])

        rep = c.num_heads // c.num_kv_heads
        keys = np.repeat(st.k, rep, axis=1)  # [L, heads, d], kv head j serves q heads j*rep .. j*rep+rep-1
        vals = np.repeat(st.v, rep, axis=1)
        scores = np.einsum("thd,lhd->htl", q, keys) / math.sqrt(c.head_dim)
        L = keys.shape[0]
        mask = np.arange(L)[None, :] > positions[:, None]  # key position after query position
        scores = np.where(mask[None], -np.inf, scores)
        scores -= scores.max(axis=-1, keepdims=True)
        p = np.exp(scores)
        p /= p.sum(axis=-1, keepdims=True)
        o = np.einsum("htl,lhd->thd", p, vals).reshape(T, -1)
        return (o * sigmoid(gate)) @ w["o_proj"].T

    def _gated_delta_net(self, w: dict, x: np.ndarray, st: LayerState) -> np.ndarray:
        c = self.cfg
        T = x.shape[0]
        kd, vd, hk, hv = c.lin_k_dim, c.lin_v_dim, c.lin_k_heads, c.lin_v_heads
        qkv = x @ w["in_proj_qkv"].T  # [T, conv_dim]
        z = (x @ w["in_proj_z"].T).reshape(T, hv, vd)
        beta = sigmoid(x @ w["in_proj_b"].T)  # [T, hv]
        g = -np.exp(w["a_log"]) * softplus(x @ w["in_proj_a"].T + w["dt_bias"])  # [T, hv]

        # Depthwise causal conv over time, continuing from the cached rows.
        K = c.conv_kernel
        padded = np.concatenate([st.conv, qkv])  # [K-1+T, C]
        conv = np.zeros_like(qkv)
        for j in range(K):
            conv += padded[j : j + T] * w["conv"][:, j]
        st.conv = padded[-(K - 1) :]
        conv = silu(conv)

        q = conv[:, : hk * kd].reshape(T, hk, kd)
        k = conv[:, hk * kd : 2 * hk * kd].reshape(T, hk, kd)
        v = conv[:, 2 * hk * kd :].reshape(T, hv, vd)
        if hv != hk:
            q = np.repeat(q, hv // hk, axis=1)
            k = np.repeat(k, hv // hk, axis=1)
        q = l2_norm(q) / math.sqrt(kd)
        k = l2_norm(k)

        s = st.s
        out = np.empty((T, hv, vd), np.float32)
        for t in range(T):
            s = s * np.exp(g[t])[:, None, None]
            kv_mem = np.einsum("hkv,hk->hv", s, k[t])
            delta = (v[t] - kv_mem) * beta[t][:, None]
            s = s + k[t][:, :, None] * delta[:, None, :]
            out[t] = np.einsum("hkv,hk->hv", s, q[t])
        st.s = s

        # Gated RMSNorm per head: plain weight (not 1 + w), then silu(z).
        normed = out / np.sqrt(np.mean(out * out, axis=-1, keepdims=True) + c.eps) * w["norm"]
        return (normed * silu(z)).reshape(T, -1) @ w["out_proj"].T

    # -- model ----------------------------------------------------------------

    def forward(self, ids: list[int], state: State, trace: dict | None = None) -> np.ndarray:
        """Appends `ids` to `state`; returns logits [len(ids), vocab]. `trace` collects per-layer hidden states."""
        c = self.cfg
        h = self.embed[np.asarray(ids)]
        if trace is not None:
            trace["embed"] = h.copy()
        for i, (kind, w, st) in enumerate(zip(c.layer_types, self.layers, state.layers, strict=True)):
            xn = rms_norm(h, w["in_norm"], c.eps)
            if kind == "full_attention":
                h = h + self._attention(w, xn, st, state.pos)
            else:
                h = h + self._gated_delta_net(w, xn, st)
            if trace is not None:
                trace[f"layer{i:02d}.mixer"] = h.copy()
            xn = rms_norm(h, w["post_norm"], c.eps)
            h = h + (silu(xn @ w["gate"].T) * (xn @ w["up"].T)) @ w["down"].T
            if trace is not None:
                trace[f"layer{i:02d}.out"] = h.copy()
        state.pos += len(ids)
        assert h.dtype == np.float32, h.dtype  # NumPy 2 promotes to f64 on any np.float64 scalar
        h = rms_norm(h, self.final_norm, c.eps)
        logits = h @ self.embed.T
        if trace is not None:
            trace["final_norm"] = h.copy()
            trace["logits"] = logits.copy()
        return logits

    def greedy(self, prompt_ids: list[int], max_new: int, state: State | None = None) -> list[int]:
        state = state or self.new_state()
        logits = self.forward(prompt_ids, state)
        out: list[int] = []
        for _ in range(max_new):
            nxt = int(np.argmax(logits[-1]))
            out.append(nxt)
            if nxt == self.cfg.eos_token_id:
                break
            logits = self.forward([nxt], state)
        return out
