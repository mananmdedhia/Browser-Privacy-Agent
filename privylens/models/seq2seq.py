"""Encoder-decoder Transformer: reads a sentence, WRITES the sanitised version token by token.
(Comparison model: we expect it to damage text that should be left alone.)

  encoder:  src ids -> N x EncoderBlock                              (same blocks as the tagger)
  decoder:  previous output tokens + encoder output -> N x DecoderBlock -> scores over the vocabulary
Training uses teacher forcing: the decoder is fed the CORRECT previous tokens and must predict the next.
The output layer re-uses the embedding table (weight tying), as in the original Transformer.
"""
import torch
import torch.nn as nn
from privylens.models.layers import (DecoderBlock, Embedding, EncoderBlock, PositionalEncoding,
                                     causal_mask, pad_mask)
from privylens.tokenizer.bpe import BOS, EOS, PAD


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, d=128, heads=4, d_ff=256, enc_layers=2, dec_layers=2, dropout=0.1):
        super().__init__()
        self.emb, self.pos, self.drop = Embedding(vocab_size, d), PositionalEncoding(d), nn.Dropout(dropout)
        self.enc = nn.ModuleList(EncoderBlock(d, heads, d_ff, dropout) for _ in range(enc_layers))
        self.dec = nn.ModuleList(DecoderBlock(d, heads, d_ff, dropout) for _ in range(dec_layers))

    def encode(self, src):
        mask = pad_mask(src)
        x = self.drop(self.pos(self.emb(src)))
        for blk in self.enc:
            x = blk(x, mask)
        return x, mask

    def decode(self, tgt_in, enc, src_mask):
        self_mask = causal_mask(tgt_in.size(1), tgt_in.device) & pad_mask(tgt_in)   # no future, no padding
        x = self.drop(self.pos(self.emb(tgt_in)))
        for blk in self.dec:
            x = blk(x, enc, self_mask, src_mask)
        return x @ self.emb.emb.weight.T                  # tied output layer -> (B,T,vocab)

    def forward(self, src, tgt_in):
        enc, mask = self.encode(src)
        return self.decode(tgt_in, enc, mask)

    @torch.no_grad()
    def generate(self, src, max_len=130):
        """Greedy decoding: always pick the single most likely next token, until [EOS]."""
        enc, mask = self.encode(src)
        out = torch.full((src.size(0), 1), BOS, device=src.device)
        done = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            nxt = self.decode(out, enc, mask)[:, -1].argmax(-1).masked_fill(done, PAD)   # finished rows just pad
            out = torch.cat([out, nxt[:, None]], 1)
            done |= nxt == EOS
            if done.all():
                break
        return out[:, 1:]


def load_seq2seq(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    model = Seq2Seq(**ck["config"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval()
