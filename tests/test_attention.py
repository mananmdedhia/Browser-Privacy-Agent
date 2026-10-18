import torch
import torch.nn as nn
from privylens.models.layers import (EncoderBlock, MultiHeadAttention, PositionalEncoding,
                                     causal_mask, pad_mask)

torch.manual_seed(0)
D, H = 16, 4


def test_shapes_and_attention_rows_sum_to_one():
    m = MultiHeadAttention(D, H).eval()
    x = torch.randn(2, 5, D)
    assert m(x).shape == (2, 5, D)
    assert m.attn.shape == (2, H, 5, 5)
    assert torch.allclose(m.attn.sum(-1), torch.ones(2, H, 5))


def test_matches_pytorch_reference():
    # Our attention must give the same numbers as PyTorch's built-in one (used here only as a checker).
    ours = MultiHeadAttention(D, H).eval()
    ref = nn.MultiheadAttention(D, H, batch_first=True).eval()
    with torch.no_grad():
        ref.in_proj_weight.copy_(torch.cat([ours.q.weight, ours.k.weight, ours.v.weight]))
        ref.in_proj_bias.copy_(torch.cat([ours.q.bias, ours.k.bias, ours.v.bias]))
        ref.out_proj.weight.copy_(ours.o.weight); ref.out_proj.bias.copy_(ours.o.bias)
        x = torch.randn(2, 6, D)
        assert torch.allclose(ours(x), ref(x, x, x)[0], atol=1e-5)


def test_padding_is_ignored():
    # Changing what sits in the padding slots must not change the real tokens' outputs.
    blk = EncoderBlock(D, H, 32).eval()
    ids = torch.tensor([[5, 6, 7, 0, 0]])
    x1, x2 = torch.randn(1, 5, D), torch.randn(1, 5, D)
    x2[:, :3] = x1[:, :3]                           # same real tokens, different junk in the pad slots
    mask = pad_mask(ids)
    assert torch.allclose(blk(x1, mask)[:, :3], blk(x2, mask)[:, :3], atol=1e-5)
    m = MultiHeadAttention(D, H).eval(); m(x1, mask=mask)
    assert m.attn[..., 3:].abs().max() == 0         # zero attention paid to pad keys


def test_causal_mask_hides_future():
    m = MultiHeadAttention(D, H).eval()
    x1 = torch.randn(1, 5, D); x2 = x1.clone(); x2[:, 3:] = torch.randn(1, 2, D)   # only the future differs
    mask = causal_mask(5)
    assert torch.allclose(m(x1, mask=mask)[:, :3], m(x2, mask=mask)[:, :3], atol=1e-5)


def test_positional_encoding():
    x = torch.zeros(1, 4, D)
    pe = PositionalEncoding(D)(x)[0]
    assert torch.allclose(pe[0, 0::2], torch.zeros(D // 2)) and torch.allclose(pe[0, 1::2], torch.ones(D // 2))  # pos 0: sin=0, cos=1
    assert PositionalEncoding(D, learned=True)(x).shape == x.shape
