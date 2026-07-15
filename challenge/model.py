"""nano-gpt — the target agents optimize.

`generate_ids(prompt, n_new)` greedily decodes `n_new` tokens from a real GPT-style
transformer. The reference below is intentionally NAIVE: at every decode step it
recomputes the full forward pass over the whole sequence from scratch (no KV cache).

An agent's job: make decoding dramatically faster while returning the EXACT same
token ids. The classic win is a KV cache (reuse past keys/values instead of
recomputing them) — mathematically identical, so the output tokens are unchanged.
The arena checks the output on a hidden holdout, so you can't fake it.
"""
import numpy as np

from _gpt import CFG, P, gelu, ln, softmax


def forward(ids):
    """Full forward over the whole sequence -> logits (T, vocab)."""
    ne, nh = CFG["n_embd"], CFG["n_head"]
    hd = ne // nh
    T = len(ids)
    x = P["wte"][ids] + P["wpe"][:T]
    mask = np.triu(np.full((T, T), -1e10), 1)
    for b in P["blocks"]:
        a = ln(x, b["ln1_g"], b["ln1_b"])
        qkv = a @ b["attn_w"] + b["attn_b"]
        q, k, v = np.split(qkv, 3, axis=-1)

        def sh(z):
            return z.reshape(T, nh, hd).transpose(1, 0, 2)

        q, k, v = sh(q), sh(k), sh(v)
        att = softmax(q @ k.transpose(0, 2, 1) / np.sqrt(hd) + mask)
        o = (att @ v).transpose(1, 0, 2).reshape(T, ne)
        x = x + o @ b["proj_w"] + b["proj_b"]
        a2 = ln(x, b["ln2_g"], b["ln2_b"])
        h = gelu(a2 @ b["fc_w"] + b["fc_b"])
        x = x + h @ b["fcp_w"] + b["fcp_b"]
    x = ln(x, P["lnf_g"], P["lnf_b"])
    return x @ P["wte"].T


def generate_ids(prompt, n_new):
    ids = list(prompt)
    for _ in range(n_new):
        logits = forward(ids)
        ids.append(int(np.argmax(logits[-1])))
    return ids[len(prompt):]
