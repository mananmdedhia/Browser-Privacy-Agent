# Results (running log)

All detection numbers are exact-match spans. Leaked = private info left visible (lower is better).
Data: ai4privacy (English) + our synthetic Indian data. See `CHECKLIST.md` for how each was produced.

## Overall (span detection: regex, our tagger, DistilBERT)

| Model | Params | Split | Precision | Recall | F1 | Leaked spans | Leaked chars | Over-redaction |
|---|---|---|---|---|---|---|---|---|
| Regex baseline | - | test | 0.723 | 0.229 | 0.348 | 71.6% | 59.2% | 0.56% |
| Regex baseline | - | unseen templates | 1.000 | 0.485 | 0.653 | 51.5% | 46.4% | 0.00% |
| **Encoder tagger** (from scratch, 25 epochs) | 1.29M | test | 0.815 | 0.899 | **0.855** | 5.7% | 3.0% | 0.77% |
| **Encoder tagger** | 1.29M | unseen templates | 0.674 | 0.768 | **0.718** | 15.4% | 6.3% | 3.19% |
| DistilBERT (pretrained, fine-tuned 4 epochs) | 65.2M (~261 MB) | test | 0.947 | 0.966 | **0.956** | 2.3% | 1.9% | 0.40% |
| DistilBERT | 65.2M | unseen templates | 0.925 | 0.960 | **0.942** | 0.4% | 0.1% | 1.07% |
| **Hybrid engine** (regex + 2-layer tagger, merged, spans snapped to whole words) | 1.29M | test | 0.875 | 0.935 | **0.904** | 3.7% | 2.1% | 1.20% |
| **Hybrid engine** (same) | 1.29M | unseen templates | 0.776 | 0.872 | **0.821** | 8.5% | 3.8% | 2.76% |

Reading it: our 1.29M-parameter model is 50x smaller than DistilBERT. It trails by about 10 F1 points on `test`
and about 22 on unseen templates, but beats regex by a wide margin on both.

## Per-entity F1 on test

| Entity | Regex | Our tagger | DistilBERT |
|---|---|---|---|
| PERSON | 0.00 | 0.83 | 0.94 |
| LOCATION | 0.00 | 0.89 | 0.96 |
| ORGANIZATION | 0.00 | 0.98 | 1.00 |
| EMAIL | 0.98 | 0.87 | 0.97 |
| PHONE | 0.51 | 0.80 | 0.93 |
| DATE | 0.58 | 0.91 | 0.96 |
| ID_NUMBER | 0.12 | 0.82 | 0.95 |
| IP_ADDRESS | 0.90 | 0.84 | 0.97 |
| AADHAAR | 0.44 | 0.99 | 1.00 |
| PAN | 1.00 | 1.00 | 1.00 |
| URL | 1.00 | 1.00 | 1.00 |

## Encoder-decoder (seq2seq): generating the sanitised text instead of tagging it

Scored only on rows that fit in 128 tokens: 3441/3993 of `test` (86%), 595/595 of unseen templates.
Different subset from the tagger table, so do NOT compare the two directly. Placeholder F1 counts the right
placeholders in any position (more forgiving than span F1).

| Split | Exact match | Placeholder F1 | Leaked (verbatim) | Preservation | Hallucinated | Dropped | Text changed |
|---|---|---|---|---|---|---|---|
| test | 23.0% | 0.874 | 1.6% | 0.864 | 27.3% | 28.0% | 35.0% |
| unseen templates | 28.7% | 0.785 | 2.2% | 0.862 | 18.0% | 56.3% | 12.4% |

Columns: Hallucinated / Dropped / Text changed = share of rows with an invented placeholder / a missing
placeholder / altered ordinary words. Seq2seq: 1.69M params, 15 epochs, greedy decoding.

**Failure modes seen in the examples (the evidence for "tagging beats generation")**
- Duplicated words: "Episode Transcript Transcript."
- Corrupted ordinary words: "Geail" instead of the next word.
- Partial copy of private text instead of a placeholder: `jagish [EMAIL]`, `daten [EMAIL]` (the leak metric only
  catches the full value copied verbatim, so true leakage is higher than the 1.6-2.2% shown).
- Wrong placeholder type: an email became `[ID_NUMBER]`, a person became `[ORGANIZATION]`.
- Complete breakdown on unseen sentence shapes: "Witness statement of [ORGANIZATION] of Mangated, dangated."
- Only 1 in 4 outputs is exactly right, while a tagger cannot alter the text at all.

## Ablations on the encoder tagger (one change at a time)

Baseline = 2 layers, 4 heads, sinusoidal positions, 8k vocab (25 epochs). Every ablation ran 15 epochs, 1 seed each.
For fairness the baseline's val F1 at epoch 15 was 0.860, so the 15-epoch variants are not meaningfully handicapped.

| Variant | Params | Test F1 | Test leaked | Unseen F1 | Unseen leaked | Val F1 (best) |
|---|---|---|---|---|---|---|
| **Baseline** (2L, 4 heads, sin, 8k) | 1.29M | 0.855 | 5.7% | 0.718 | 15.4% | 0.876 |
| **4 layers** | 1.56M | **0.887** | **4.3%** | **0.797** | **7.0%** | **0.908** |
| 4 layers, 25 epochs (same budget as baseline) | 1.56M | 0.910 | 3.8% | 0.809 | 7.0% | 0.930 |
| 6 layers, 25 epochs | 1.82M | **0.922** | **3.4%** | **0.821** | 8.1% | 0.932 |
| 2 heads | 1.29M | 0.827 | 7.4% | 0.657 | 27.3% | 0.839 |
| 8 heads | 1.29M | 0.812 | 7.6% | 0.647 | 24.9% | 0.828 |
| Learned positions | 1.36M | 0.865 | 5.7% | 0.618 | 17.2% | 0.875 |
| Vocab 4k | 0.78M | 0.850 | 6.6% | 0.682 | 14.1% | 0.854 |
| Vocab 16k | 2.32M | 0.807 | 7.4% | 0.634 | 22.8% | 0.820 |

Findings
- **Depth is the biggest win:** 4 layers beat 2 on every number (+3.2 F1 on test, +7.9 on unseen, leakage on unseen 15.4% -> 7.0%) for +20% parameters.
- **Depth, equal 25-epoch budget:** 2 layers 0.855 -> 4 layers 0.910 -> 6 layers 0.922 test F1 (unseen 0.718 -> 0.809 -> 0.821).
  The jump from 2 to 4 layers (+5.5 F1) is much bigger than from 4 to 6 (+1.2 F1 for +17% params): diminishing returns.
  Unseen-template leakage is lowest at 4 layers (7.0%) vs 8.1% at 6, so 4 layers is the chosen production size
  (smaller and faster for the browser, same quality within noise).
- **Heads:** 4 is best; both 2 and 8 are worse, mostly on unseen templates (d_model=128: 8 heads leaves only 16 dims per head).
- **Positions:** learned ~ sinusoidal on `test` (0.865 vs 0.855) but clearly worse on unseen templates (0.618 vs 0.718):
  consistent with learned positions memorising template layouts, while fixed sin/cos generalises.
- **Vocabulary:** 4k matches 8k on `test` (0.850 vs 0.855) with 40% fewer parameters (the embedding table is most of the model); 16k is worse (rare tokens under-trained).
- Caveats: single seed, and the unseen set is only 595 rows from 5 templates, so differences of a few points there are within noise. Trust the trends (depth helps, 16k hurts), not the decimals.

## Case robustness (found by a user trying an ALL-CAPS name)

Problem: `MANAN MIHIR RAYCHAND DEDHIA` was not detected. Cause: only 41 of 17,772 PERSON examples in train were
ALL CAPS and 50 lowercase, so the model had effectively never seen a capitalised name. Fixes:
1. Training data: random UPPER (15%) and lower (10%) case copies added to train only (val/test/unseen byte-identical,
   so old and new models stay comparable); tokenizer retrained on it (`tokenizers/bpe_8k_caps.json`).
2. New split `test_caps`: 1,997 rows = upper- and lower-case copies of 1,000 test rows.
3. Engine: model spans are widened to whole words (`snap`), so half a word is never left visible.

| Model on `test_caps` (before the retrain) | F1 | Left visible | PERSON recall |
|---|---|---|---|
| 2-layer tagger alone | 0.462 | 40.1% | 0.08 |
| 2-layer hybrid + snap | 0.616 | 27.6% | 0.13 |
| 4-layer tagger retrained with case augmentation | pending (Colab) | pending | pending |

Word snapping alone, on the old model and the normal test sets: hybrid test F1 0.862 -> 0.904, left visible 5.3% -> 3.7%,
unseen F1 0.733 -> 0.821, left visible 15.7% -> 8.5%. Half-word fragments were a large share of all errors.
Still open: lowercase PAN is missed by regex (case-sensitive pattern).

## Hybrid engine notes

- Merge rule: precise regex (email, URL, IP, PAN) > neural tagger > loose regex (phone, Aadhaar, date, ID, kept only
  where the model found nothing). First version ("longest span wins, regex wins ties") scored WORSE than the tagger
  alone (test F1 0.823): loose regex labels overrode correct model labels (Aadhaar precision 0.40, phone 0.42).
- Gains over the tagger alone are modest (+0.7 F1 test, +1.5 unseen); unseen-template EMAIL recall 0.71 -> 1.00.
  Remaining leakage is mostly PERSON / LOCATION misses by the model, which regex cannot help with.
- End-to-end check: 25 of 1,304 gold values (1.9%) still visible verbatim in sanitised output on 300 test rows.
- Known quirk: the model tags some pronouns as PERSON ("His employee ID" -> "[PERSON_1] employee ID").
- ONNX export (`checkpoints/tagger.onnx`, 5.5 MB) matches PyTorch to 7.6e-06.

## Notes (read before quoting these numbers)

- Tagger val F1 plateaus near 0.87 from epoch ~20 while train loss keeps falling: mild overfitting.
- PAN / URL / AADHAAR / ORGANIZATION come only from our template-generated data, so the 0.98-1.00 scores on `test`
  partly reflect template familiarity. `test_unseen_templates` is the honest generalisation number.
- The unseen set has no ORGANIZATION examples, so its ORGANIZATION 0.00 (tagger and DistilBERT) is pure false positives, not a miss.
- Unseen-template EMAIL is weak for both neural models (0.58 tagger, 0.73 DistilBERT) while regex scores 1.00:
  the case for the hybrid engine (Section 8).
- Embedding-init bug fixed before the tagger run (the first run, F1 0.21, had it).
- Still to run: latency benchmark; hybrid numbers with the 4-layer tagger; CRF ablation (optional).
