"""Fixed model internals — NOT editable by agents.

Holds the (seeded) weights, config, and elementwise primitives. Agents optimize
the *decode loop* in model.py; the weights and math live here so every candidate
runs the exact same model and its outputs are comparable.
"""
import numpy as np

CFG = dict(vocab=256, n_embd=128, n_head=4, n_layer=4, block=256)


def _init():
    rng = np.random.default_rng(1234)
    C = CFG
    ne, V, L = C["n_embd"], C["vocab"], C["n_layer"]

    def r(*shape, sc=0.06):
        return (rng.standard_normal(shape) * sc).astype(np.float64)

    P = dict(wte=r(V, ne, sc=0.12), wpe=r(C["block"], ne, sc=0.02),
             lnf_g=np.ones(ne), lnf_b=np.zeros(ne), blocks=[])
    for _ in range(L):
        P["blocks"].append(dict(
            ln1_g=np.ones(ne), ln1_b=np.zeros(ne),
            attn_w=r(ne, 3 * ne), attn_b=np.zeros(3 * ne),
            proj_w=r(ne, ne), proj_b=np.zeros(ne),
            ln2_g=np.ones(ne), ln2_b=np.zeros(ne),
            fc_w=r(ne, 4 * ne), fc_b=np.zeros(4 * ne),
            fcp_w=r(4 * ne, ne), fcp_b=np.zeros(ne)))
    return P


P = _init()


def ln(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def gelu(x):
    return 0.5 * x * (1 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def softmax(x, axis=-1):
    m = x.max(axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis, keepdims=True)
