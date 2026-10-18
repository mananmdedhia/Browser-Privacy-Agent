"""Our own Byte-Pair Encoding tokenizer (no libraries).

Idea: start with single characters; repeatedly merge the most frequent
neighbouring pair into a new token ("R"+"a" -> "Ra") until the vocab is full.
Common words become one token, rare strings (IDs, names) split into pieces.

Train:  python -m privylens.tokenizer.bpe 8000     -> tokenizers/bpe_8k.json
        (trains on the TRAIN split only)
"""
import json
import sys
from collections import Counter, defaultdict
from privylens.tokenizer.pretokenize import MARK, PLACEHOLDERS, pretokenize

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[BOS]", "[EOS]"]
PAD, UNK, CLS, SEP, BOS, EOS = range(6)
# placeholders exist in two forms: plain and "after a space"
FIXED = SPECIALS + PLACEHOLDERS + [MARK + p for p in PLACEHOLDERS]


def _merge(word, a, b):
    """Replace every adjacent (a, b) in the symbol list with the joined symbol."""
    out, i = [], 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
            out.append(a + b); i += 2
        else:
            out.append(word[i]); i += 1
    return out


class BPE:
    def __init__(self, vocab, merges):
        self.vocab = vocab                                  # token string -> id
        self.inv = {i: t for t, i in vocab.items()}
        self.merges = [tuple(m) for m in merges]            # in the order they were learned
        self.ranks = {m: i for i, m in enumerate(self.merges)}
        self.cache = {}

    # ---------- training ----------
    @classmethod
    def train(cls, texts, vocab_size):
        freq = Counter(w for t in texts for w, _, _ in pretokenize(t)
                       if w.lstrip(MARK) not in PLACEHOLDERS)
        words = [list(w) for w in freq]                      # each word as a list of symbols
        fr = list(freq.values())
        vocab = {t: i for i, t in enumerate(FIXED)}
        for ch in sorted({c for w in freq for c in w}):
            vocab.setdefault(ch, len(vocab))

        pairs, where = Counter(), defaultdict(set)           # pair counts; pair -> words containing it
        def count(i, sign):
            for p in zip(words[i], words[i][1:]):
                pairs[p] += sign * fr[i]
                if sign > 0: where[p].add(i)
                elif pairs[p] <= 0: del pairs[p]
        for i in range(len(words)):
            count(i, +1)

        merges = []
        while len(vocab) < vocab_size and pairs:
            best = max(pairs, key=pairs.get)
            if pairs[best] < 2: break                        # nothing worth merging
            merges.append(best)
            vocab[best[0] + best[1]] = len(vocab)
            for i in list(where.pop(best)):                  # only touch words that contain the pair
                count(i, -1)
                words[i] = _merge(words[i], *best)
                count(i, +1)
        return cls(vocab, merges)

    # ---------- using it ----------
    def _word_symbols(self, word):
        if word in self.cache: return self.cache[word]
        if word.lstrip(MARK) in PLACEHOLDERS:
            syms = [word]
        else:
            syms = list(word)
            while len(syms) > 1:
                cand = [(self.ranks[p], p) for p in zip(syms, syms[1:]) if p in self.ranks]
                if not cand: break
                syms = _merge(syms, *min(cand)[1])           # apply the earliest-learned merge
        self.cache[word] = syms
        return syms

    def encode(self, text):
        """-> (ids, offsets). offsets[i] = (char_start, char_end) of token i in `text`."""
        ids, offsets = [], []
        for word, start, _ in pretokenize(text):
            pos = start
            for sym in self._word_symbols(word):
                n = len(sym) - (1 if sym.startswith(MARK) else 0)   # marker is not a real character
                ids.append(self.vocab.get(sym, UNK))
                offsets.append((pos, pos + n))
                pos += n
        return ids, offsets

    def decode(self, ids):
        skip = {PAD, CLS, SEP, BOS, EOS}
        return "".join(self.inv[i] for i in ids if i not in skip).replace(MARK, " ")

    def save(self, path):
        json.dump({"vocab": self.vocab, "merges": self.merges}, open(path, "w", encoding="utf-8"), ensure_ascii=False)

    @classmethod
    def load(cls, path):
        d = json.load(open(path, encoding="utf-8"))
        return cls(d["vocab"], d["merges"])


if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    texts = [json.loads(l)["text"] for l in open("data/splits/train.jsonl", encoding="utf-8")]
    tok = BPE.train(texts, size)
    path = f"tokenizers/bpe_{size // 1000}k.json"
    tok.save(path)
    print(f"saved {path}: vocab={len(tok.vocab)}, merges={len(tok.merges)}")
    ids, _ = tok.encode("Send the report to Rahul Sharma at rahul@example.com.")
    print([tok.inv[i] for i in ids])
