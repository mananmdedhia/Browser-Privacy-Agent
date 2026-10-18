import torch
import torch.nn.functional as F
from privylens.data.seq2seq_pairs import make_pair
from privylens.models.seq2seq import Seq2Seq
from privylens.tokenizer.bpe import BOS, EOS, PAD


def test_make_pair():
    text = "Send to Rahul Sharma at rahul@example.com."
    spans = [{"start": 8, "end": 20, "label": "PERSON"}, {"start": 24, "end": 41, "label": "EMAIL"}]
    assert make_pair(text, spans)[1] == "Send to [PERSON] at [EMAIL]."


def test_decoder_cannot_see_future():
    torch.manual_seed(0)
    m = Seq2Seq(50, d=32, heads=4, d_ff=64, dropout=0.0).eval()
    src = torch.randint(6, 50, (1, 7))
    a = torch.randint(6, 50, (1, 6)); b = a.clone(); b[:, 4:] = torch.randint(6, 50, (1, 2))   # only the end differs
    assert torch.allclose(m(src, a)[:, :4], m(src, b)[:, :4], atol=1e-5)


def test_learns_and_generates():
    # a working encoder-decoder can memorise 4 random (source -> target) pairs and reproduce them
    torch.manual_seed(0)
    m = Seq2Seq(40, d=32, heads=4, d_ff=64, enc_layers=2, dec_layers=2, dropout=0.0)
    src = torch.randint(6, 40, (4, 6)); tgt = torch.randint(6, 40, (4, 5))
    tgt_in = torch.cat([torch.full((4, 1), BOS), tgt], 1); tgt_out = torch.cat([tgt, torch.full((4, 1), EOS)], 1)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(300):
        loss = F.cross_entropy(m(src, tgt_in).transpose(1, 2), tgt_out, ignore_index=PAD)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    assert m.generate(src, max_len=8)[:, :6].tolist() == tgt_out.tolist()
