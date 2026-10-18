# Results (running log)

All numbers are exact-match spans. Leaked = private info left visible (lower is better).
Data: ai4privacy (English) + our synthetic Indian data. See `CHECKLIST.md` for how each was produced.

## Overall

| Model | Split | Precision | Recall | F1 | Leaked spans | Leaked chars | Over-redaction |
|---|---|---|---|---|---|---|---|
| Regex baseline | test | 0.723 | 0.229 | 0.348 | 71.6% | 59.2% | 0.56% |
| Regex baseline | unseen templates | 1.000 | 0.485 | 0.653 | 51.5% | 46.4% | 0.00% |
| **Encoder tagger** (1.29M params, 25 epochs, Colab T4) | test | 0.815 | 0.899 | **0.855** | **5.7%** | 3.0% | 0.77% |
| **Encoder tagger** | unseen templates | 0.674 | 0.768 | **0.718** | 15.4% | 6.3% | 3.19% |

## Per-entity F1 on test

| Entity | Regex | Tagger |
|---|---|---|
| PERSON | 0.00 | 0.83 |
| LOCATION | 0.00 | 0.89 |
| ORGANIZATION | 0.00 | 0.98 |
| EMAIL | 0.98 | 0.87 |
| PHONE | 0.51 | 0.80 |
| DATE | 0.58 | 0.91 |
| ID_NUMBER | 0.12 | 0.82 |
| IP_ADDRESS | 0.90 | 0.84 |
| AADHAAR | 0.44 | 0.99 |
| PAN | 1.00 | 1.00 |
| URL | 1.00 | 1.00 |

## Notes (read before quoting these numbers)

- Tagger val F1 plateaus near 0.87 from epoch ~20 while train loss keeps falling: mild overfitting, more epochs will not help.
- PAN / URL / AADHAAR / ORGANIZATION come only from our template-generated data, so the 0.98-1.00 scores on `test` partly reflect template familiarity. `test_unseen_templates` is the honest generalisation number.
- The unseen set has no ORGANIZATION examples, so its ORGANIZATION 0.00 is pure false positives, not a miss.
- Regex still beats the tagger on EMAIL, IP and unseen-template EMAIL: this is the case for the hybrid engine (Section 8).
- Embedding-init bug fixed before this run (first run, F1 0.21, was with the bug).
