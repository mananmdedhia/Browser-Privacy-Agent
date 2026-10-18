"""Draw the tagger architecture for the presentation, with the REAL parameter counts
(the numbers come from building the actual model, not typed in by hand).

Run:  python -m privylens.evaluation.model_summary                      (default sizes)
      python -m privylens.evaluation.model_summary checkpoints/tagger.pt (sizes of a trained model)
Out:  reports/figures/tagger_architecture.png  + a parameter table printed to the console
"""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import FancyBboxPatch
from privylens.models.encoder_tagger import EncoderTagger
from privylens.schema import TAGS

DEFAULT = dict(vocab_size=8000, num_tags=len(TAGS), d=128, heads=4, d_ff=256, layers=2, dropout=0.1, learned_pos=False)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def draw(cfg, path="reports/figures/tagger_architecture.png"):
    m = EncoderTagger(**cfg)
    d, N = cfg["d"], cfg["layers"]
    fig, ax = plt.subplots(figsize=(7, 11))
    ax.set_xlim(0, 10); ax.set_ylim(0, 22); ax.axis("off")

    def box(y, h, title, detail="", color="#e3f2fd", x=1.5, w=7, dashed=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
                                    fc=color, ec="#455a64", lw=1.4, ls="--" if dashed else "-"))
        ax.text(x + w / 2, y + h / 2 + (0.18 if detail else 0), title, ha="center", va="center", fontsize=11, weight="bold")
        if detail:
            ax.text(x + w / 2, y + h / 2 - 0.27, detail, ha="center", va="center", fontsize=9, color="#37474f")

    def arrow(y0, y1):
        ax.annotate("", xy=(5, y1), xytext=(5, y0), arrowprops=dict(arrowstyle="->", color="#455a64", lw=1.5))

    blk = n_params(m.blocks[0])
    pos = "learned" if cfg["learned_pos"] else "sinusoidal"
    # bottom -> top
    box(0.2, 1.1, "Input text", "Send the report to Rahul Sharma at rahul@example.com.", "#eceff1")
    arrow(1.4, 2.0)
    box(2.1, 1.1, "BPE tokenizer (built from scratch)", f"{cfg['vocab_size']:,}-token vocabulary  ->  ids  (B, T)", "#fff3e0")
    arrow(3.3, 3.9)
    box(4.0, 1.1, "Token embedding  x sqrt(d)", f"d = {d}   |   {n_params(m.emb):,} params", "#e3f2fd")
    arrow(5.2, 5.8)
    box(5.9, 1.1, "+ Positional encoding", f"{pos}   |   {n_params(m.pos):,} params", "#e3f2fd")
    arrow(7.1, 7.7)
    # encoder stack: container + four inner boxes
    ax.add_patch(FancyBboxPatch((1.5, 7.8), 7, 8.3, boxstyle="round,pad=0.05,rounding_size=0.15",
                                fc="#f1f8e9", ec="#455a64", lw=1.4, ls="--"))
    ax.text(5, 15.75, f"Encoder block  x {N}", ha="center", va="center", fontsize=11, weight="bold")   # label at top of container
    ax.text(5, 15.35, f"{blk:,} params per block  ({N * blk:,} total)", ha="center", va="center", fontsize=9, color="#37474f")
    ys = [8.4, 10.1, 11.8, 13.5]
    parts = [("Multi-head self-attention", f"{cfg['heads']} heads, padding mask", "#c8e6c9"),
             ("Add & LayerNorm", "residual connection", "#dcedc8"),
             ("Feed-forward", f"{d} -> {cfg['d_ff']} -> {d}, GELU", "#c8e6c9"),
             ("Add & LayerNorm", "residual connection", "#dcedc8")]
    for y, (t, dtl, c) in zip(ys, parts):
        box(y, 1.3, t, dtl, c, x=2.2, w=5.6)
    for a, b in zip(ys, ys[1:]):
        arrow(a + 1.3, b)
    arrow(16.2, 16.8)
    box(16.9, 1.1, "Linear head", f"{d} -> {cfg['num_tags']} tags   |   {n_params(m.head):,} params", "#fce4ec")
    arrow(18.1, 18.7)
    box(18.8, 1.1, "Tag for every token (BIO)", "O, B-PERSON, I-PERSON, B-EMAIL, ...", "#f3e5f5")
    arrow(20.0, 20.5)
    box(20.6, 1.1, "Character spans -> [PERSON_1], [EMAIL_1]", "", "#eceff1")
    ax.text(5, 22, f"PrivyLens encoder tagger  -  {n_params(m):,} parameters", ha="center", fontsize=13, weight="bold")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print(f"saved {path}\n{'part':22s}{'params':>10s}")
    for name, mod in [("embedding", m.emb), ("positional", m.pos), (f"{N} encoder blocks", m.blocks), ("head", m.head), ("TOTAL", m)]:
        print(f"{name:22s}{n_params(mod):>10,}")


if __name__ == "__main__":
    cfg = torch.load(sys.argv[1], map_location="cpu")["config"] if len(sys.argv) > 1 else DEFAULT
    draw(cfg)
