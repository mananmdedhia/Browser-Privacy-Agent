# PrivyLens AI — Project Specification

**Custom Transformers for Privacy-Preserving Text Sanitisation**
ATML + NLP Minor Project

> Detect and redact personally identifiable information (PII) in free text using Transformer models built and trained from scratch, compare them against rule-based and pretrained baselines, and ship the best system as a local privacy engine with a lightweight browser demo.

---

## 1. Problem Statement

People paste emails, documents and chat logs into AI tools and online forms without noticing the names, phone numbers, ID numbers and addresses inside them. Once that text leaves the device, the user loses control of it.

PrivyLens detects sensitive spans **on the user's side** and replaces them with typed, reversible placeholders before the text is shared:

```
Input : Send the report to Rahul Sharma at rahul@example.com. His employee ID is EMP4821.
Output: Send the report to [PERSON_1] at [EMAIL_1]. His employee ID is [EMPLOYEE_ID_1].
```

The mapping `[PERSON_1] → Rahul Sharma` stays on the device, so the user can restore the original text in any response.

---

## 2. Objectives

| # | Objective | Subject focus |
|---|-----------|---------------|
| O1 | Implement a Transformer encoder (embeddings, positional encoding, multi-head self-attention, FFN, residuals, LayerNorm) from scratch in PyTorch | ATML |
| O2 | Implement a full encoder-decoder Transformer (masked self-attention, cross-attention, teacher forcing, greedy/beam decoding) | ATML |
| O3 | Train a custom BPE tokenizer and align span labels to subword tokens | NLP |
| O4 | Frame PII detection as BIO sequence labelling and evaluate at span level | NLP |
| O5 | Compare four approaches: regex, custom encoder, custom encoder-decoder, fine-tuned DistilBERT | Both |
| O6 | Build a hybrid privacy engine (regex + custom tagger) with validation and reversible placeholders | Engineering |
| O7 | Demonstrate it in a thin browser extension / web demo | Engineering |

---

## 3. Final System Structure

| # | Component | Role in the project |
|---|-----------|---------------------|
| 1 | **Regex baseline** | Rule-based detector for structured PII; lower-bound baseline |
| 2 | **Custom encoder-only Transformer tagger** (own BPE tokenizer) | **Main model** — token classification with BIO tags |
| 3 | **Custom encoder-decoder Transformer** | Comparison experiment — generates sanitised text directly |
| 4 | **Fine-tuned DistilBERT** | Reference upper bound (pretrained knowledge) |
| 5 | **Hybrid privacy engine + demo** | Regex for structured PII + custom tagger for contextual PII, validated, exposed via API and browser extension |

---

## 4. High-Level Architecture

```
                         ┌──────────────────────────────────────────┐
                         │              RESEARCH TRACK               │
                         │                                          │
   Raw datasets ───────► │  Data pipeline ──► BPE tokenizer         │
   (public + synthetic)  │        │                 │               │
                         │        ▼                 ▼               │
                         │  ┌───────────┐  ┌──────────────────┐     │
                         │  │ 1. Regex  │  │ 2. Encoder tagger │◄── main
                         │  └───────────┘  └──────────────────┘     │
                         │  ┌──────────────────┐ ┌──────────────┐   │
                         │  │ 3. Enc-Dec seq2seq│ │ 4. DistilBERT│   │
                         │  └──────────────────┘ └──────────────┘   │
                         │        │ all models                      │
                         │        ▼                                 │
                         │  Unified evaluation harness ──► Results  │
                         └───────────────┬──────────────────────────┘
                                         │ best model exported (ONNX / .pt)
                                         ▼
┌──────────────┐   text   ┌──────────────────────────────────────────────┐
│   Browser    │ ───────► │            PRIVACY ENGINE (local)             │
│  extension / │          │                                              │
│  web demo    │ ◄─────── │  Normalise → Regex detector ─┐               │
│              │ sanitised│            → Custom tagger ──┼─► Span merger  │
└──────────────┘  text +  │                              │        │      │
                  review  │                              ▼        ▼      │
                  UI      │                   Validator (re-scan output) │
                          │                              │               │
                          │                   Placeholder vault (local)  │
                          └──────────────────────────────────────────────┘
```

Everything in the privacy engine runs **on the user's machine**. No raw text is sent to any server.

---

## 5. Repository Structure

```
privylens/
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── data.yaml                 # dataset sources, splits, entity schema
│   ├── tokenizer.yaml            # vocab size, special tokens
│   ├── encoder_tagger.yaml       # model + training hyperparameters
│   ├── seq2seq.yaml
│   └── distilbert.yaml
│
├── data/
│   ├── raw/                      # downloaded public datasets (git-ignored)
│   ├── synthetic/                # generated Indian-identifier examples
│   ├── processed/                # unified JSONL: text + char-level spans
│   └── splits/                   # train / val / test / test_unseen_templates
│
├── privylens/
│   ├── __init__.py
│   ├── schema.py                 # entity label set, BIO tag maps
│   │
│   ├── data/
│   │   ├── loaders.py            # load + convert public datasets to unified format
│   │   ├── synthetic.py          # template + Faker generator (PAN, Aadhaar, phone…)
│   │   ├── splitter.py           # template-aware train/test splitting
│   │   └── seq2seq_pairs.py      # build (original, sanitised) pairs from spans
│   │
│   ├── tokenizer/
│   │   ├── bpe.py                # BPE trainer + encoder/decoder (own implementation)
│   │   ├── pretokenize.py        # whitespace/punct split that keeps @ . - _ intact
│   │   └── align.py              # char spans → subword BIO labels
│   │
│   ├── models/
│   │   ├── layers/
│   │   │   ├── embeddings.py     # token embedding + scaling
│   │   │   ├── positional.py     # sinusoidal + learned positional encodings
│   │   │   ├── attention.py      # scaled dot-product + multi-head attention
│   │   │   ├── feedforward.py    # position-wise FFN
│   │   │   ├── encoder_block.py  # MHA → Add&Norm → FFN → Add&Norm
│   │   │   └── decoder_block.py  # masked MHA → cross-attn → FFN
│   │   ├── encoder_tagger.py     # encoder stack + token classification head
│   │   ├── seq2seq.py            # full encoder-decoder + generation
│   │   └── distilbert_tagger.py  # HF wrapper (reference only)
│   │
│   ├── baselines/
│   │   └── regex_detector.py     # patterns for EMAIL, PHONE, PAN, AADHAAR, URL, IP…
│   │
│   ├── training/
│   │   ├── train_tagger.py
│   │   ├── train_seq2seq.py
│   │   ├── train_distilbert.py
│   │   ├── losses.py             # CE with ignore_index, optional class weights
│   │   └── schedulers.py         # warmup + inverse-sqrt / cosine
│   │
│   ├── evaluation/
│   │   ├── span_metrics.py       # span-level P/R/F1 per entity (seqeval)
│   │   ├── leakage.py            # leaked-span rate, leaked-character rate
│   │   ├── preservation.py       # non-sensitive text preservation for seq2seq
│   │   ├── latency.py            # inference timing (CPU / ONNX)
│   │   ├── attention_viz.py      # attention heatmaps for the report
│   │   └── run_benchmark.py      # evaluates all 4 approaches → results table
│   │
│   └── engine/
│       ├── pipeline.py           # normalise → detect → merge → validate → redact
│       ├── merger.py             # resolve overlapping regex/model spans
│       ├── validator.py          # re-scan sanitised output, flag low confidence
│       ├── vault.py              # reversible placeholder mapping (local only)
│       └── export_onnx.py        # export tagger for in-browser inference
│
├── api/
│   └── server.py                 # FastAPI: /sanitise, /restore (localhost only)
│
├── extension/                    # Chrome extension (plain JS)
│   ├── manifest.json
│   ├── popup.html / popup.js / popup.css
│   ├── content.js                # reads selected / textarea text
│   └── review.js                 # highlight spans, accept/reject, copy sanitised
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_tokenizer_analysis.ipynb
│   ├── 03_results_and_plots.ipynb
│   └── 04_attention_visualisation.ipynb
│
├── tests/
│   ├── test_attention.py         # shape + masking correctness
│   ├── test_bpe.py               # encode/decode round-trip
│   ├── test_align.py             # span → BIO alignment
│   ├── test_regex.py
│   └── test_engine.py            # end-to-end: no known PII survives
│
├── checkpoints/                  # git-ignored
└── reports/
    ├── figures/
    └── results.md                # final comparison tables
```

---

## 6. Entity Schema

Two groups. This split drives the hybrid design: regex handles the first group, the model handles the second.

| Group | Entities | Primary detector |
|-------|----------|------------------|
| **Structured** (pattern-based) | `EMAIL`, `PHONE`, `PAN`, `AADHAAR`, `URL`, `IP_ADDRESS`, `DATE`, `ID_NUMBER` (employee/case/account IDs) | Regex (model as backup) |
| **Contextual** (needs understanding) | `PERSON`, `ORGANIZATION`, `LOCATION` / `ADDRESS` | Custom tagger |

Tagging scheme: **BIO** (`B-PERSON`, `I-PERSON`, `O`, …).
Keep the label set small (~10 entities) so each class has enough training data.

---

## 7. Data Pipeline

### 7.1 Sources
1. **Public PII dataset** — e.g. the ai4privacy PII-masking datasets on Hugging Face (verify licence and label set before use). Map their labels onto the schema above.
2. **Public NER data** (optional) — CoNLL-2003 for extra `PERSON` / `ORGANIZATION` / `LOCATION` coverage.
3. **Synthetic Indian data** — template sentences filled with Faker (`en_IN` locale) plus valid-format PAN, Aadhaar-style numbers, Indian phone numbers and Indian names/cities. **All values must be fake.**

### 7.2 Unified format (`data/processed/*.jsonl`)
```json
{
  "id": "syn_000123",
  "text": "Send the report to Rahul Sharma at rahul@example.com.",
  "spans": [
    {"start": 19, "end": 31, "label": "PERSON"},
    {"start": 35, "end": 52, "label": "EMAIL"}
  ],
  "source": "synthetic",
  "template_id": "T042"
}
```
Character-level spans are the single source of truth. BIO labels and seq2seq targets are both derived from them.

### 7.3 Splits
| Split | Purpose |
|-------|---------|
| `train` | Training |
| `val` | Early stopping, hyperparameter tuning |
| `test` | Standard held-out evaluation |
| `test_unseen_templates` | Synthetic templates **never seen in training** — measures real generalisation, not template memorisation |

### 7.4 Seq2seq pairs
Generated from the same spans:
```
source: Send the report to Rahul Sharma at rahul@example.com.
target: Send the report to [PERSON] at [EMAIL].
```
(Numbered placeholders like `[PERSON_1]` are added later by the engine, not learned by the model.)

---

## 8. Tokenizer (own implementation)

- **Algorithm:** Byte-Pair Encoding, trained on the training split only.
- **Vocab size:** start at 8,000; compare 4k / 8k / 16k in an ablation.
- **Pre-tokenisation:** split on whitespace and most punctuation, but keep `@ . _ -` inside tokens so emails and IDs aren't shredded.
- **Special tokens:** `[PAD] [UNK] [CLS] [SEP] [BOS] [EOS]` plus one token per entity placeholder (`[PERSON]`, `[EMAIL]`, …) for the seq2seq target vocabulary.
- **Label alignment:** first subword of a span gets `B-X`, the rest get `I-X`. Store character offsets per token so predictions map back to the original text exactly.

---

## 9. Model Architectures

### 9.1 Model 1 — Regex baseline
Hand-written patterns for each structured entity, plus checksum/format checks where possible (e.g. PAN format `AAAAA9999A`, 12-digit Aadhaar grouping, Indian mobile prefixes). Cannot detect `PERSON` / `ORG` / `LOCATION`; that gap is part of the finding.

### 9.2 Model 2 — Custom Encoder-only Tagger (main model)

```
token ids ──► Token Embedding (d=128) × √d
          ──► + Positional Encoding (sinusoidal; learned in ablation)
          ──► Dropout
          ──► [ Encoder Block ] × N
                 ├─ Multi-Head Self-Attention (h=4, padding mask)
                 ├─ Add & LayerNorm
                 ├─ Feed-Forward (128 → 256 → 128, GELU)
                 └─ Add & LayerNorm
          ──► Linear (128 → num_BIO_tags)
          ──► (optional) CRF layer for valid BIO transitions
          ──► per-token tag + softmax confidence
```

| Hyperparameter | Starting value |
|----------------|----------------|
| Encoder layers (N) | 2 (ablate 2 / 4) |
| Attention heads | 4 |
| d_model | 128 |
| d_ff | 256 |
| Max sequence length | 128 tokens (sliding window for longer text) |
| Dropout | 0.1 |
| Optimizer | AdamW, lr 5e-4, weight decay 0.01 |
| Schedule | Linear warmup (~5% steps) → cosine decay |
| Loss | Cross-entropy, `ignore_index` on padding; optional class weights for rare entities |
| Batch size | 32–64 |

These are starting values to tune, not measured results.

### 9.3 Model 3 — Custom Encoder-Decoder (comparison)

```
Encoder: same blocks as Model 2 (N=2)
Decoder: [ Decoder Block ] × 2
           ├─ Masked Multi-Head Self-Attention (causal mask)
           ├─ Add & LayerNorm
           ├─ Multi-Head Cross-Attention over encoder output
           ├─ Add & LayerNorm
           ├─ Feed-Forward
           └─ Add & LayerNorm
         ──► Linear (128 → vocab) (weights tied with decoder embedding)
Training : teacher forcing, cross-entropy, label smoothing 0.1
Inference: greedy decoding (beam search 3–5 as an ablation)
```

**Expected failure modes to measure and discuss:** dropped words, paraphrased words, corrupted rare tokens, hallucinated placeholders. These are the evidence for why sequence labelling suits redaction better.
**Optional extension:** add a copy / pointer-generator mechanism and measure how much it closes the gap.

### 9.4 Model 4 — Fine-tuned DistilBERT (reference)
`distilbert-base-cased` + token classification head, fine-tuned on the same splits. Used **only** as a reference point, never in the engine. It answers: *how close does a from-scratch model get to a pretrained one, and at what size/latency?*

---

## 10. Evaluation Plan

### 10.1 Metrics
| Metric | What it measures | Applies to |
|--------|------------------|------------|
| Span-level Precision / Recall / F1 (per entity + micro/macro) | Detection quality (exact span match, `seqeval`) | All |
| **Leaked-span rate** = missed spans / total spans | Privacy risk. Recall is the primary privacy metric | All |
| Leaked-character rate | Partial leaks (e.g. half a name left visible) | All |
| Over-redaction rate | Non-sensitive text wrongly masked (usability) | All |
| Text preservation (exact match / chrF on non-sensitive tokens) | Did generation keep the rest of the text intact? | Model 3 |
| Parameters, model size (MB) | Efficiency | Models 2–4 |
| Latency (ms / 100 tokens, CPU + ONNX) | On-device feasibility | Models 2–4, hybrid |

### 10.2 Results tables to produce
1. Overall comparison of all 4 approaches + hybrid (micro F1, recall, leakage, latency, size).
2. **Per-entity F1** (structured vs contextual), showing where regex wins and where the model wins.
3. `test` vs `test_unseen_templates`: the generalisation gap.
4. Ablations on the main model: layers (2/4), heads (2/4/8), positional encoding (sinusoidal/learned), vocab size, with/without CRF.
5. Seq2seq error analysis: categorised examples of its failure modes.

### 10.3 Qualitative
- Attention heatmaps showing what the tagger attends to when labelling a name (e.g. "to", "Mr.", "at").
- Confusion matrix across entity types.

---

## 11. Hybrid Privacy Engine

```
text
 │
 ├─► normalise (unicode, whitespace) — keep char offset map
 │
 ├─► Regex detector ─────────► structured spans (high confidence)
 ├─► Custom tagger ──────────► contextual spans + confidence
 │
 ├─► Span merger
 │     • overlapping spans → prefer longer span; regex wins for structured types
 │     • union of both detectors (favour recall: missing PII is worse than over-masking)
 │
 ├─► Redactor → numbered placeholders, consistent per value
 │     "Rahul Sharma" appears twice → [PERSON_1] both times
 │
 ├─► Validator
 │     • re-run regex on the sanitised output; any hit = leak → block
 │     • spans with model confidence below threshold → flag for user review
 │
 └─► Vault (local only): { "[PERSON_1]": "Rahul Sharma", ... }
       /restore swaps placeholders back into an AI response
```

**API (localhost only)**
| Endpoint | Input | Output |
|----------|-------|--------|
| `POST /sanitise` | `{ text }` | `{ sanitised_text, spans[], flagged[], session_id }` |
| `POST /restore` | `{ text, session_id }` | `{ restored_text }` |

**Deployment options for the tagger**
- A: Local FastAPI server running the PyTorch model (simplest).
- B: Export to ONNX and run in the browser with `onnxruntime-web`. Fully client-side, no server at all. A strong privacy talking point, and feasible because the model is small.

---

## 12. Browser Extension (thin demo)

- Select text or focus a textarea → click **Sanitise**.
- Popup shows the text with detected spans highlighted by type; the user can toggle any span on/off.
- Flagged (low-confidence) spans are shown in a different colour.
- **Copy sanitised** button; **Restore** button to paste back an AI reply with real values reinserted.
- Built in plain JavaScript/HTML/CSS (Manifest V3). Talks to `localhost` (option A) or runs the ONNX model directly (option B).

The extension is built **last** and kept minimal. The research track is the core of the project.

---

## 13. Tech Stack

| Layer | Tools |
|-------|-------|
| Models & training | Python 3.11, PyTorch (custom modules only for Models 2–3) |
| Reference model | Hugging Face `transformers` (Model 4 only) |
| Data | `datasets`, `Faker` (en_IN), pandas |
| Evaluation | `seqeval`, scikit-learn, matplotlib / seaborn |
| Experiment tracking | Weights & Biases or TensorBoard |
| Serving | FastAPI, uvicorn, ONNX / onnxruntime(-web) |
| Extension | Plain JS, HTML, CSS (Manifest V3) |
| Testing | pytest |

**Rule:** `privylens/models/` must not import any pretrained Transformer for Models 2 and 3. Only `distilbert_tagger.py` may use `transformers`.

---

## 14. Timeline (≈10 weeks)

| Week | Milestone |
|------|-----------|
| 1 | Finalise schema, collect public data, write unified converter, build synthetic generator, create splits |
| 2 | Regex baseline + evaluation harness (span metrics, leakage) — first numbers on the board |
| 3 | BPE tokenizer + label alignment + tests |
| 4 | Attention, MHA, encoder block from scratch + unit tests |
| 5 | Train encoder tagger; tune; first comparison vs regex |
| 6 | Decoder block + full seq2seq; train; error analysis |
| 7 | Fine-tune DistilBERT reference; run ablations on main model |
| 8 | Hybrid engine: merger, validator, vault, FastAPI; ONNX export |
| 9 | Browser extension demo; latency benchmarks |
| 10 | Final results, attention visualisations, report, presentation |

Weeks 1–7 are the must-haves. If time runs short, cut the extension down to a simple web page, not the experiments.

---

## 15. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| From-scratch model generalises poorly to unseen names | Mix public + synthetic data; report the gap honestly against DistilBERT; hybrid design covers structured PII anyway |
| Inflated scores from template memorisation | `test_unseen_templates` split; template-aware splitting |
| Seq2seq output corrupts text | Expected. It's an experiment, not the production path; measure and analyse it |
| Rare entity classes (e.g. `ADDRESS`) underperform | Class-weighted loss, extra synthetic examples, report per-entity |
| Subword/label misalignment bugs | Unit tests on alignment; round-trip span reconstruction check |
| Scope creep from the extension | Built last, minimal; web page fallback |
| Using real personal data | Only public datasets and Faker-generated values; never real PII in the repo |

---

## 16. Expected Contributions

1. A from-scratch Transformer encoder and encoder-decoder implementation for PII detection, with attention visualisations.
2. A four-way empirical comparison (rule-based, custom encoder, custom seq2seq, pretrained) showing where each approach succeeds and fails.
3. Evidence on why sequence labelling suits redaction better than generation.
4. A small, on-device hybrid privacy engine with validation and reversible placeholders, demonstrated in a browser extension.

---

## 17. Inspiration

SIH26190 — *Secure Digital Document Management System for Legal and Investigation Documents.* PrivyLens addresses one piece of that problem: sanitising sensitive information in documents and text before they are shared or processed by AI tools.
