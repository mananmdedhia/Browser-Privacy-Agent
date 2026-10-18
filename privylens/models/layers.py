"""Transformer building blocks, written from scratch (no nn.MultiheadAttention,
no nn.TransformerEncoder). Only basic PyTorch pieces are used: Linear, LayerNorm,
Embedding, Dropout. Shapes: B = batch, T = tokens, d = model width, h = heads.

Pieces, in the order data flows through them:
  Embedding            token id  -> vector of size d
  PositionalEncoding   adds "where in the sentence am I" to each vector
  MultiHeadAttention   every token looks at every other token and mixes in what's useful
  FeedForward          per-token two-layer network (extra thinking power)
  EncoderBlock         attention + feed-forward, each wrapped in residual + LayerNorm
Masks (True = allowed to look at):
  pad_mask     stop attention to [PAD] filler tokens
  causal_mask  stop a token seeing the future (needed by the decoder in Section 6)
"""
import math
import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, vocab_size, d, pad_id=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d, padding_idx=pad_id)
        self.scale = math.sqrt(d)               # keeps embeddings comparable in size to positions
        # small start values (std d^-0.5): after the sqrt(d) scale they are ~1, same size as the position signal.
        # PyTorch's default (std 1) made tokens ~11x louder than positions and saturated attention.
        nn.init.normal_(self.emb.weight, std=d ** -0.5)
        with torch.no_grad():
            self.emb.weight[pad_id].zero_()

    def forward(self, ids):                      # (B,T) -> (B,T,d)
        return self.emb(ids) * self.scale


class PositionalEncoding(nn.Module):
    """Attention has no sense of word order, so we add a position signal.
    sinusoidal: fixed sin/cos waves of different speeds. learned: a trainable table (ablation)."""
    def __init__(self, d, max_len=512, learned=False):
        super().__init__()
        self.learned = learned
        if learned:
            self.pe = nn.Embedding(max_len, d)
        else:
            pos = torch.arange(max_len).unsqueeze(1)
            freq = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
            pe = torch.zeros(max_len, d)
            pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
            self.register_buffer("pe", pe)       # saved with the model but not trained

    def forward(self, x):                        # (B,T,d) -> (B,T,d)
        T = x.size(1)
        pe = self.pe(torch.arange(T, device=x.device)) if self.learned else self.pe[:T]
        return x + pe


def pad_mask(ids, pad_id=0):
    """(B,T) ids -> (B,1,1,T) bool: False where the key is padding."""
    return (ids != pad_id)[:, None, None, :]


def causal_mask(T, device=None):
    """(1,1,T,T) bool: token i may look at tokens 0..i only."""
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))[None, None]


class MultiHeadAttention(nn.Module):
    """Split d into h smaller "heads"; each head learns its own way of looking at the sentence.
    Self-attention: MHA(x). Cross-attention (decoder): MHA(x, kv=encoder_output)."""
    def __init__(self, d, heads, dropout=0.1):
        super().__init__()
        assert d % heads == 0, "d must divide evenly into heads"
        self.h, self.dk = heads, d // heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.drop = nn.Dropout(dropout)
        self.attn = None                          # last attention weights, kept for heatmaps

    def forward(self, x, kv=None, mask=None):
        kv = x if kv is None else kv
        B, Tq, Tk = x.size(0), x.size(1), kv.size(1)
        split = lambda t, T: t.view(B, T, self.h, self.dk).transpose(1, 2)       # (B,h,T,dk)
        q, k, v = split(self.q(x), Tq), split(self.k(kv), Tk), split(self.v(kv), Tk)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dk)                    # (B,h,Tq,Tk): how much q cares about k
        if mask is not None:                      # blocked positions get ~-infinity -> 0 after softmax
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        w = scores.softmax(-1)
        self.attn = w.detach()
        out = (self.drop(w) @ v).transpose(1, 2).reshape(B, Tq, -1)             # merge heads back
        return self.o(out)


class FeedForward(nn.Module):
    def __init__(self, d, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d))

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    """attention -> add & norm -> feed-forward -> add & norm. ("add" = residual: keep the input too)"""
    def __init__(self, d, heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn, self.ff = MultiHeadAttention(d, heads, dropout), FeedForward(d, d_ff, dropout)
        self.n1, self.n2, self.drop = nn.LayerNorm(d), nn.LayerNorm(d), nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.n1(x + self.drop(self.attn(x, mask=mask)))
        return self.n2(x + self.drop(self.ff(x)))


class DecoderBlock(nn.Module):
    """The decoder writes the output one token at a time. Each block has three steps:
    1. masked self-attention: look at what has been written so far (never the future)
    2. cross-attention: look at the encoder's reading of the input sentence
    3. feed-forward
    each wrapped in residual + LayerNorm."""
    def __init__(self, d, heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn, self.cross_attn = MultiHeadAttention(d, heads, dropout), MultiHeadAttention(d, heads, dropout)
        self.ff = FeedForward(d, d_ff, dropout)
        self.n1, self.n2, self.n3, self.drop = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d), nn.Dropout(dropout)

    def forward(self, x, enc, self_mask=None, cross_mask=None):
        x = self.n1(x + self.drop(self.self_attn(x, mask=self_mask)))
        x = self.n2(x + self.drop(self.cross_attn(x, kv=enc, mask=cross_mask)))
        return self.n3(x + self.drop(self.ff(x)))
