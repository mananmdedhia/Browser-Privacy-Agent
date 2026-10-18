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
| **Hybrid engine** (regex + tagger, merged) | 1.29M | test | 0.820 | 0.908 | **0.862** | 5.3% | 2.7% | 1.05% |
| **Hybrid engine** | 1.29M | unseen templates | 0.677 | 0.798 | **0.733** | 15.7% | 5.9% | 2.35% |

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
- Still to run: ablations (layers, heads, position type, vocab size), latency benchmark.
