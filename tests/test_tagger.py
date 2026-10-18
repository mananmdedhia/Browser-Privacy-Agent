import torch
import torch.nn.functional as F
from privylens.models.encoder_tagger import EncoderTagger


def test_shape_and_tiny_batch_is_learnable():
    torch.manual_seed(0)
    model = EncoderTagger(vocab_size=50, num_tags=5, d=32, heads=4, d_ff=64, layers=2, dropout=0.0)
    ids = torch.randint(1, 50, (4, 12))
    tags = torch.randint(0, 5, (4, 12))
    assert model(ids).shape == (4, 12, 5)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(300):                            # a working model can memorise 4 random sequences
        loss = F.cross_entropy(model(ids).transpose(1, 2), tags)
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.1
