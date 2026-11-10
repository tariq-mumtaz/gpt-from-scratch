import json 
import numpy as np
from load_weights import load_safetensors

def layer_norm(x, weight, bias, eps=1e-5):
    # Works on each row (token) separately: axis=-1 = across the row,
    # keepdims=True = return one value per row as a column, so it lines up with x.

    # 1. Mean: add up the row's numbers, divide by how many there are
    mean = x.mean(axis=-1, keepdims=True)

    # 2 + 3. Variance: subtract the mean from each number, square each result,
    #        add them up, divide by how many there are (x.var does both steps)
    var = x.var(axis=-1, keepdims=True)

    # 4. Standard deviation: square root of the variance
    #    (eps = tiny number so we never divide by zero)
    std = np.sqrt(var + eps)

    # 5. Normalize: subtract the mean and divide by std -> each row has mean 0, spread 1
    norm = (x - mean) / std

    # 6. Adjust with learned values: scale by weight, shift by bias (768 each)
    return norm * weight + bias

def softmax(x):
    # Subtract each row's largest value first. The result is the same,
    # but it stops np.exp from overflowing on large scores.
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)                                   # step 1: e to the power of each score
    return e / e.sum(axis=-1, keepdims=True)        # step 2: divide by the row's total

def attention_head(q, k, v):
    # q, k, v: (tokens, 64) for one head
    n = q.shape[0]
    scores = q @ k.T / np.sqrt(q.shape[1])             # 4b: compare every query with every key
    allowed = np.tril(np.ones((n, n)))                 # 4c: only self and earlier tokens
    scores = np.where(allowed == 1, scores, -np.inf)
    weights = softmax(scores)                          # 4d: scores -> weights summing to 1
    return weights @ v                                 # 4e: weighted mix of the values

def gelu(x):
    # GPT-2's GELU (the tanh version): lets positive values through,
    # squashes negative values towards 0, with a smooth curve in between
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def attention(x, weights, prefix):
    # Multi-head causal self-attention (Step 4), for one block
    n_heads = 12
    d = x.shape[-1]                  # 768
    head_size = d // n_heads         # 64

    qkv = x @ weights[prefix + "attn.c_attn.weight"] + weights[prefix + "attn.c_attn.bias"]
    q = qkv[:, 0:d]
    k = qkv[:, d:2 * d]
    v = qkv[:, 2 * d:3 * d]

    heads = []
    for i in range(n_heads):
        start = i * head_size
        end = start + head_size
        heads.append(attention_head(q[:, start:end], k[:, start:end], v[:, start:end]))
    attn = np.concatenate(heads, axis=-1)

    return attn @ weights[prefix + "attn.c_proj.weight"] + weights[prefix + "attn.c_proj.bias"]

def mlp(x, weights, prefix):
    # Feed-forward (Step 5): expand to 3072, GELU, shrink back to 768
    up = gelu(x @ weights[prefix + "mlp.c_fc.weight"] + weights[prefix + "mlp.c_fc.bias"])
    return up @ weights[prefix + "mlp.c_proj.weight"] + weights[prefix + "mlp.c_proj.bias"]

def block(x, weights, i):
    # One transformer block: layer norm + attention + residual, then layer norm + MLP + residual
    prefix = f"h.{i}."               # e.g. "h.0." -> names like "h.0.ln_1.weight"
    h = layer_norm(x, weights[prefix + "ln_1.weight"], weights[prefix + "ln_1.bias"])
    x = x + attention(h, weights, prefix)
    h = layer_norm(x, weights[prefix + "ln_2.weight"], weights[prefix + "ln_2.bias"])
    x = x + mlp(h, weights, prefix)
    return x

def gpt2(ids, weights, n_layers=12):
    # Step 2: token + position embeddings
    x = weights["wte.weight"][ids] + weights["wpe.weight"][:len(ids)]

    # Steps 3-5: the same block, 12 times, each with its own weights
    for i in range(n_layers):
        x = block(x, weights, i)
        # Step 7: final layer norm, then compare with every token's embedding
    x = layer_norm(x, weights["ln_f.weight"], weights["ln_f.bias"])
    return x @ weights["wte.weight"].T       # (tokens, 50257) scores, called "logits"


if __name__ == "__main__":
    weights = load_safetensors("models/gpt2/model.safetensors")
    vocab = json.load(open("models/gpt2/vocab.json"))
    id_to_token = {i: t for t, i in vocab.items()}       # reverse: ID -> token

    words = ["The", "Ġcapital", "Ġof", "ĠFrance", "Ġis"]
    ids = [vocab[w] for w in words]
    print("ids:", ids)

    logits = gpt2(ids, weights)              # (5, 50257)
    probs = softmax(logits[-1])              # last token's scores -> probabilities

    top5 = np.argsort(probs)[::-1][:5]       # IDs of the 5 highest probabilities
    for t in top5:
        print(repr(id_to_token[t]), round(float(probs[t]), 3))