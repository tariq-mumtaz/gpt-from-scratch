import regex
import json

# GPT-2's rule for cutting text into words, before any merging
SPLIT_PATTERN = regex.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

def split_words(text):
    return SPLIT_PATTERN.findall(text)

def bytes_to_unicode():
    # Every possible byte (0-255) gets one visible character,
    # so vocab.json can store any token as readable text
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    mapping = {}
    extra = 0
    for b in range(256):
        if b in visible:
            mapping[b] = chr(b)              # already visible: keep its own character
        else:
            mapping[b] = chr(256 + extra)    # invisible (space, newline...): use a stand-in
            extra += 1
    return mapping

def load_merges(path):
    # read merges.txt into a dictionary: (left, right) -> priority (0 = highest)
    with open(path,"r",encoding="utf-8") as f:
        lines = f.read().split("\n")[1:]            #skip the version line
        
    ranks = {}
    for i,line in enumerate(lines):
        if line:
            left,right = line.split(" ")
            ranks[(left,right)] = i

    return ranks

def bpe(word,ranks):
    # Start with single characters, then keep gluing the highest-priority neighbour pair
    pieces = list(word)
    while len(pieces) > 1:
        # 1. Find the neighbour pair with the best (lowest) rank
        best_pair = None
        for i in range(len(pieces)-1):
            pair = (pieces[i],pieces[i+1])
            if pair in ranks:
                if best_pair is None or ranks[pair] < ranks[best_pair]:
                    best_pair = pair
        # 2. no pair has a rule : done
        if best_pair is None:
            break

        # 3. Glue every occurrence of that pair
        merged=[]
        i=0
        while i < len(pieces):
            if i < len(pieces) - 1 and (pieces[i],pieces[i+1]) == best_pair:
                merged.append(pieces[i] + pieces[i+1])
                i += 2
            else:
                merged.append(pieces[i])
                i += 1
        pieces = merged
    return pieces

class Tokenizer:
    def __init__(self,vocab_path,merges_path):
        self.vocab = json.load(open(vocab_path,"r",encoding="utf-8"))
        self.id_to_token = {i : t for t, i in self.vocab.items()}
        self.ranks = load_merges(merges_path)
        self.byte_to_char = bytes_to_unicode()
        self.char_to_byte = {c: b for b, c in self.byte_to_char.items()}

    def encode(self,text):
        # text --> list to token items
        ids = []
        for word in split_words(text):
            chars = ""
            for b in word.encode("utf-8"):
                chars += self.byte_to_char[b]
            for piece in bpe(chars, self.ranks):
                ids.append(self.vocab[piece])
        return ids

    def decode(self, ids):
        # list of token ids --> text (the same step backward)
        chars = ""
        for i in ids:
            chars += self.id_to_token[i]
        data = bytes([self.char_to_byte[c] for c in chars])
        return data.decode("utf-8",errors="replace")
    

if __name__ == "__main__":
    tok = Tokenizer("models/gpt2/vocab.json", "models/gpt2/merges.txt")

    ids = tok.encode("The capital of France is")
    print(ids)
    print(tok.decode(ids))

    print(tok.encode("Hello world"))

    arabic = "مرحبا"
    ids_ar = tok.encode(arabic)
    print(len(ids_ar), "tokens:", ids_ar)
    print(tok.decode(ids_ar))