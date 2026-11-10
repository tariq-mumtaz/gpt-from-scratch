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

#something learned from block 0's learned lin_1 values
ln1_w = weights["h.0.ln_1.weight"]
ln1_b = weights["h.0.ln_1.bias"]

print(f"ln1_w weight : {ln1_w[:5]}")
print(f"ln1_b bias : {ln1_b[:5]}")

h = layer_norm(x, ln1_w, ln1_b)
print(f"h shape : {h.shape}")