import json
import struct

path = "models/gpt2/model.safetensors"

with open(path,"rb") as f:
    header_len = struct.unpack("<Q",f.read(8))[0]
    header = json.loads(f.read(header_len))

print("header length :", header_len, " bytes")
print("entries: " , len(header))

for name in ["wte.weight", "h.0.attn.c_attn.weight","h.0.attn.bias"]:
    print(name, header.get(name))