from load_weights import load_safetensors

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