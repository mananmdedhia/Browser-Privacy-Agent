"""The main model: a Transformer encoder that labels every token (O / B-PERSON / I-EMAIL ...).

   token ids -> Embedding -> + positions -> dropout -> N x EncoderBlock -> Linear -> one score per tag

predict() is the helper everything else uses (evaluation now, the privacy engine later):
text in, character spans out.
"""
import torch
import torch.nn as nn
from privylens.models.layers import Embedding, EncoderBlock, PositionalEncoding, pad_mask
from privylens.tokenizer.align import to_spans


class EncoderTagger(nn.Module):
    def __init__(self, vocab_size, num_tags, d=128, heads=4, d_ff=256, layers=2, dropout=0.1, learned_pos=False):
        super().__init__()
        self.emb, self.pos = Embedding(vocab_size, d), PositionalEncoding(d, learned=learned_pos)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(EncoderBlock(d, heads, d_ff, dropout) for _ in range(layers))
        self.head = nn.Linear(d, num_tags)

    def forward(self, ids):                      # (B,T) token ids -> (B,T,num_tags) scores
        mask = pad_mask(ids)
        x = self.drop(self.pos(self.emb(ids)))
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(x)


def load_tagger(path, device="cpu"):
    ck = torch.load(path, map_location=device)   # {"model": weights, "config": constructor args}
    model = EncoderTagger(**ck["config"])
    model.load_state_dict(ck["model"])
    model.tokenizer_path = ck.get("tokenizer", "tokenizers/bpe_8k.json")   # older checkpoints used the 8k tokenizer
    return model.to(device).eval()


@torch.no_grad()
def predict(model, tok, text, window=128, device="cpu"):
    """text -> list of {start, end, label}. Long text is cut into overlapping 128-token
    windows; where windows overlap, their probabilities are averaged."""
    model.eval()
    ids, offs = tok.encode(text)
    n = len(ids)
    if n == 0:
        return []
    starts = sorted({*range(0, max(n - window, 0) + 1, window // 2), max(n - window, 0)})   # last window reaches the end
    probs = torch.zeros(n, model.head.out_features)
    seen = torch.zeros(n, 1)
    for s in starts:
        chunk = torch.tensor([ids[s:s + window]], device=device)
        p = model(chunk).softmax(-1)[0].cpu()
        probs[s:s + len(p)] += p
        seen[s:s + len(p)] += 1
    probs = probs / seen
    spans = to_spans(offs, probs.argmax(-1).tolist())
    conf = probs.max(-1).values                       # how sure the model was about each token
    for s in spans:                                   # span confidence = its least-sure token (flagging uses this)
        inside = [c.item() for (a, b), c in zip(offs, conf) if a >= s["start"] and b <= s["end"]]
        s["conf"] = min(inside) if inside else 0.0
    return spans
