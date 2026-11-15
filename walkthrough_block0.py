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



weights = load_safetensors("models/gpt2/model.safetensors")

wte = weights["wte.weight"]

hello = wte[15496]

print(f"wte shape : {wte.shape}")
print(f"hello shape : {hello.shape}")
print(f"first 5: {hello[:5]}")


ids = [15496,995]
tok = wte[ids]
print(f"tok shape : {tok.shape}")
print(f"same row ? {(tok[0] == hello).all()}")

wpe = weights["wpe.weight"]
pos = wpe[:len(ids)]

print(f"wpe shape : {wpe.shape}")
print(f"pos shape : {pos.shape}")

x = tok + pos
print(f"x shape : {x.shape}")
print(f"tok[0][:3] : {tok[0][:3]}")
print(f"pos[0][:3] : {pos[0][:3]}")
print(f"x[0][:3] : {x[0][:3]}")

test = np.array([[1.0, 2.0, 3.0, 4.0]])
print("test:", layer_norm(test, np.ones(4), np.zeros(4)))

#check: with weight = 1 and bias = 0 every row should have mean 0 and spread 1
check = layer_norm(x,np.ones(768),np.zeros(768))
print(f"check mean: {check.mean(axis=-1)}")
print(f"check std: {check.std(axis=-1)}")

#something learned from block 0's learned ln_1 values
ln1_w = weights["h.0.ln_1.weight"]
ln1_b = weights["h.0.ln_1.bias"]

print(f"ln1_w weight : {ln1_w[:5]}")
print(f"ln1_b bias : {ln1_b[:5]}")

h = layer_norm(x, ln1_w, ln1_b)
print(f"h shape : {h.shape}")

# --- Attention, block 0 ---

# 4a. One linear layer makes Q, K and V together: 768 -> 2304 (= 3 x 768)
qkv_weight = weights["h.0.attn.c_attn.weight"]   # (768, 2304), OpenAI calls it c_attn
qkv_bias = weights["h.0.attn.c_attn.bias"]       # (2304,)
qkv = h @ qkv_weight + qkv_bias                  # (2, 2304)

# Split the 2304 columns into three parts of 768
q = qkv[:, 0:768]       # query: what each token is looking for
k = qkv[:, 768:1536]    # key:   what each token contains
v = qkv[:, 1536:2304]   # value: the information each token passes on

print("qkv shape:", qkv.shape)
print("q shape:  ", q.shape)

# 4b. Scores for head 0: how well each token's query matches each token's key
head_size = 64                  # 768 / 12 heads
q0 = q[:, 0:head_size]          # (2, 64) head 0's queries
k0 = k[:, 0:head_size]          # (2, 64) head 0's keys

scores = q0 @ k0.T / np.sqrt(head_size)   # (2, 2)
print("scores:\n", scores)

# 4c. Mask: a token may only look at itself and earlier tokens, never later ones
n = len(ids)                            # number of tokens, here 2
allowed = np.tril(np.ones((n, n)))      # 1 = may look, 0 = may not
scores = np.where(allowed == 1, scores, -np.inf)

print("allowed:\n", allowed)
print("masked scores:\n", scores)

# 4d. Softmax: scores -> weights that are positive and add up to 1 per row
attn_weights = softmax(scores)
print("attention weights:\n", attn_weights)
print("row sums:", attn_weights.sum(axis=-1))

# 4e. Each token's output = weighted mix of the values it may look at
v0 = v[:, 0:head_size]          # (2, 64) head 0's values
head0 = attn_weights @ v0       # (2, 2) @ (2, 64) -> (2, 64)
print("head0 shape:", head0.shape)

# Check row 1, first number, by hand
by_hand = 0.95400722 * v0[0][0] + 0.04599278 * v0[1][0]
print("matrix:  ", head0[1][0])
print("by hand: ", by_hand)


# 4f. Run all 12 heads, each on its own 64 columns, then join the results
n_heads = 12
heads = []
for i in range(n_heads):
    start = i * head_size
    end = start + head_size
    heads.append(attention_head(q[:, start:end], k[:, start:end], v[:, start:end]))

attn = np.concatenate(heads, axis=-1)    # 12 x (2, 64) -> (2, 768)
print("attn shape:", attn.shape)
print("head 0 same as before?", np.allclose(heads[0], head0))


# 4g. Output layer: mix the 12 heads' results together (768 -> 768)
proj_weight = weights["h.0.attn.c_proj.weight"]   # (768, 768), OpenAI calls it c_proj
proj_bias = weights["h.0.attn.c_proj.bias"]       # (768,)
attn_out = attn @ proj_weight + proj_bias         # (2, 768)

# Residual: add attention's result to x instead of replacing x
x1 = x + attn_out
print("x1 shape:", x1.shape)
print("x[1][:3]:        ", x[1][:3])
print("attn_out[1][:3]: ", attn_out[1][:3])
print("x1[1][:3]:       ", x1[1][:3])

# --- MLP, block 0 ---

# 5b. Layer norm, expand to 3072, GELU, shrink back to 768, then add to x1
h2 = layer_norm(x1, weights["h.0.ln_2.weight"], weights["h.0.ln_2.bias"])

up_weight = weights["h.0.mlp.c_fc.weight"]       # (768, 3072), OpenAI calls it c_fc
up_bias = weights["h.0.mlp.c_fc.bias"]           # (3072,)
down_weight = weights["h.0.mlp.c_proj.weight"]   # (3072, 768), OpenAI calls it c_proj
down_bias = weights["h.0.mlp.c_proj.bias"]       # (768,)

up = gelu(h2 @ up_weight + up_bias)              # (2, 3072): expand, then bend
mlp_out = up @ down_weight + down_bias           # (2, 768): shrink back

# Residual: add the MLP's result to x1
x2 = x1 + mlp_out
print("up shape:  ", up.shape)
print("x2 shape:  ", x2.shape)

# 6a. The functions must give exactly the same result as the step-by-step code
print("block function matches x2?", np.allclose(block(x, weights, 0), x2))