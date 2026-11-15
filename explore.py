import json
import numpy as np
from load_weights import load_safetensors

weights = load_safetensors("models/gpt2/model.safetensors")
wte = weights["wte.weight"]
vocab = json.load(open("models/gpt2/vocab.json","r",encoding="utf-8"))

def cosine(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

cat = wte[vocab["Ġcat"]]
dog = wte[vocab["Ġdog"]]
the = wte[vocab["Ġthe"]]

king = wte[vocab["Ġking"]]
queen = wte[vocab["Ġqueen"]]

india = wte[vocab["ĠIndia"]]

sword = wte[vocab["Ġsword"]]
love = wte[vocab["Ġlove"]]


print(f"cat vs dog : {cosine(cat,dog)}")
print(f"cat vs the : {cosine(cat,the)}")
print(f"king vs queen : {cosine(king,queen)}")
print(f"cat vs india : {cosine(cat,india)}")
print(f"sword vs love : {cosine(sword,love)}")