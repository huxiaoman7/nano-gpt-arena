# strategy: KV cache (legit, big win, IDENTICAL output).
# Instead of recomputing the whole prefix each step, cache each layer's keys/values
# and only process the one new token. Mathematically identical -> same argmax tokens.
import numpy as np

from _gpt import CFG, P, gelu, ln, softmax


def _run(tok_ids, start, caches):
    ne, nh = CFG["n_embd"], CFG["n_head"]
    hd = ne // nh
    T = len(tok_ids)
    x = P["wte"][tok_ids] + P["wpe"][start:start + T]
    for bi, b in enumerate(P["blocks"]):
        a = ln(x, b["ln1_g"], b["ln1_b"])
        qkv = a @ b["attn_w"] + b["attn_b"]
        q, k, v = np.split(qkv, 3, axis=-1)

        def sh(z):
            return z.reshape(T, nh, hd).transpose(1, 0, 2)

        q, k, v = sh(q), sh(k), sh(v)
        c = caches[bi]
        if c["k"] is None:
            c["k"], c["v"] = k, v
        else:
            c["k"] = np.concatenate([c["k"], k], axis=1)
            c["v"] = np.concatenate([c["v"], v], axis=1)
        K, V = c["k"], c["v"]
        L = K.shape[1]
        qpos = np.arange(start, start + T)[:, None]
        kpos = np.arange(L)[None, :]
        m = np.where(kpos <= qpos, 0.0, -1e10)
        att = softmax(q @ K.transpose(0, 2, 1) / np.sqrt(hd) + m)
        o = (att @ V).transpose(1, 0, 2).reshape(T, ne)
        x = x + o @ b["proj_w"] + b["proj_b"]
        a2 = ln(x, b["ln2_g"], b["ln2_b"])
        h = gelu(a2 @ b["fc_w"] + b["fc_b"])
        x = x + h @ b["fcp_w"] + b["fcp_b"]
    x = ln(x, P["lnf_g"], P["lnf_b"])
    return x @ P["wte"].T


def generate_ids(prompt, n_new):
    ids = list(prompt)
    caches = [{"k": None, "v": None} for _ in P["blocks"]]
    seqlen = len(ids)
    logits = _run(ids, 0, caches)
    out = []
    for i in range(n_new):
        out.append(int(np.argmax(logits[-1])))
        if i == n_new - 1:
            break
        logits = _run([out[-1]], seqlen, caches)
        seqlen += 1
    return out
