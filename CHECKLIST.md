# PrivyLens — Build Checklist

Build one item at a time, top to bottom. Tick when done + tested.
🖥️ = local (CPU is fine) · ☁️ = Google Colab (GPU) · Colab flow: push repo to GitHub → `git clone` in Colab → train → save checkpoint to Drive / download.

## 0. Setup
- [x] Repo skeleton, `requirements.txt`, `.gitignore` (data/raw, checkpoints) 🖥️
- [x] `privylens/schema.py` — entity list, BIO tag ↔ id maps 🖥️
- [ ] `configs/*.yaml` (only what training scripts actually read) 🖥️
- [x] `notebooks/colab_setup.ipynb` — clone repo, `pip install`, mount Drive, check GPU ☁️

## 1. Data (Week 1)
- [x] `data/loaders.py` — public PII dataset (ai4privacy; check licence) → unified JSONL 🖥️
- [ ] (optional) CoNLL-2003 loader for PERSON/ORG/LOC 🖥️
- [x] `data/synthetic.py` — templates + Faker `en_IN` (PAN, Aadhaar, phone, names, cities) 🖥️
- [x] `data/splitter.py` — train / val / test / `test_unseen_templates` 🖥️
- [ ] Data exploration notebook (label counts, lengths) 🖥️

## 2. Regex baseline + eval harness (Week 2)
- [x] `baselines/regex_detector.py` + `tests/test_regex.py` 🖥️
- [x] `evaluation/metrics.py` (span P/R/F1 per entity; replaces span_metrics.py, no seqeval needed) 🖥️
- [x] leakage metrics (inside metrics.py) (leaked-span, leaked-char, over-redaction) 🖥️
- [x] First regex numbers on test + unseen_templates 🖥️

## 3. Tokenizer (Week 3)
- [x] `tokenizer/pretokenize.py` (keeps `@ . _ -` inside tokens) 🖥️
- [x] `tokenizer/bpe.py` — own BPE train/encode/decode + `tests/test_bpe.py` 🖥️
- [x] `tokenizer/align.py` — char spans → subword BIO + offsets + `tests/test_align.py` 🖥️
- [x] Train BPE on train split only: 8k done; 4k / 16k later for ablation 🖥️

## 4. Transformer layers from scratch (Week 4)
- [x] embeddings + positional encoding (sinusoidal + learned), all in `models/layers.py` 🖥️
- [x] attention in layers.py (scaled dot-product + MHA, padding/causal masks) + `tests/test_attention.py` 🖥️
- [x] feedforward + encoder block in layers.py 🖥️

## 5. Encoder tagger — main model (Week 5)
- [x] `models/encoder_tagger.py` 🖥️
- [x] loss + warmup-cosine schedule (inside train_tagger.py, no separate files) 🖥️
- [x] `training/train_tagger.py` — tiny overfit sanity check (10 examples → loss ≈ 0) 🖥️
- [x] Full training run + checkpoint to Drive ☁️
- [x] Evaluate vs regex (overall, per-entity, unseen templates) 🖥️
- [ ] (optional) CRF layer ☁️

## 6. Encoder-decoder seq2seq (Week 6)
- [x] `data/seq2seq_pairs.py` 🖥️
- [x] decoder block (in layers.py) (masked self-attn + cross-attn) 🖥️
- [x] `models/seq2seq.py` — teacher forcing, greedy decode (beam as ablation) 🖥️
- [x] `training/train_seq2seq.py` written + overfit-checked; [ ] full Colab run ☁️
- [x] `evaluation/preservation.py` written; [ ] error analysis after Colab run + failure-mode error analysis 🖥️
- [ ] (optional) copy/pointer mechanism ☁️

## 7. DistilBERT reference + ablations (Week 7)
- [x] `training/train_distilbert.py` written + tested (model wrapper lives in it); [ ] Colab run ☁️
- [ ] Ablations: layers 2/4, heads 2/4/8, pos-enc, vocab size, CRF ☁️
- [ ] `evaluation/run_benchmark.py` — all approaches → results table 🖥️
- [ ] `evaluation/latency.py` + params/size 🖥️

## 8. Hybrid engine + API (Week 8)
- [ ] `engine/merger.py` 🖥️
- [ ] `engine/vault.py` (numbered, consistent placeholders) 🖥️
- [ ] `engine/validator.py` 🖥️
- [ ] `engine/pipeline.py` + `tests/test_engine.py` (no known PII survives) 🖥️
- [ ] `api/server.py` — `/sanitise`, `/restore` 🖥️
- [ ] `engine/export_onnx.py` 🖥️

## 9. Demo (Week 9)
- [ ] Extension: `manifest.json`, `content.js`, popup, `review.js` (or plain web page fallback) 🖥️

## 10. Wrap-up (Week 10)
- [ ] `evaluation/attention_viz.py` + attention notebook 🖥️
- [ ] Results + plots notebook, `reports/results.md` 🖥️
- [ ] README, report, presentation 🖥️
