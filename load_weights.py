import json
import struct
import numpy as np

MODEL_PATH = "models/gpt2/model.safetensors"

def load_safetensors(path):
    with open(path,"rb") as f:
        header_len = struct.unpack("<Q",f.read(8))[0]
        header = json.loads(f.read(header_len))
        data = f.read()

    tensors = {}
    for name,info in header.items():
        if name == "__metadata__":
            continue
        
        if info['dtype'] != 'F32':
            raise ValueError(f"{name} : dtype only allowed F32 found {info['dtype']}")
        
        start,end = info["data_offsets"]
        tensors[name] = np.frombuffer(data[start:end],dtype="<f4").reshape(info["shape"])
    return tensors

if __name__ == "__main__":
    weights = load_safetensors(MODEL_PATH)
    total = 0
    for name, t in sorted(weights.items()):
        print(f"{name:28s} {str(t.shape):18s}")
        total += t.size
    print(f"\n{len(weights)} tensors, {total:,} values")
