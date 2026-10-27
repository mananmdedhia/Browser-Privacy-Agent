"""Local-only HTTP wrapper around the privacy engine, for the browser extension.

Run:  python -m uvicorn api.server:app --host 127.0.0.1 --port 8000
      (127.0.0.1 = this computer only; text never leaves the machine)
POST /sanitise  {text, session_id?}        -> {sanitised_text, spans, flagged, blocked, leaks, session_id}
POST /categories {name, examples[], not_examples[]?} -> teach a new secret type from a few examples (kept in categories.json)
POST /restore   {text, session_id}         -> {restored_text}
POST /compare   {text}                     -> {regex, ner, privylens, distilbert}  (distilbert is null until checkpoints/distilbert.pt exists)
Uses the trained tagger if checkpoints/tagger.pt exists, otherwise regex only.
"""
import json
import os
from functools import lru_cache
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from privylens.engine.pipeline import Engine, redact

app = FastAPI(title="PrivyLens local engine")
app.add_middleware(CORSMiddleware, allow_origin_regex=r"chrome-extension://.*|http://(localhost|127\.0\.0\.1)(:\d+)?",
                   allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"])
engine = Engine.from_checkpoint() if os.path.exists("checkpoints/tagger.pt") else Engine()


CATS = "categories.json"                                     # name -> {examples, not_examples}; local file, re-learned at start-up
if os.path.exists(CATS) and os.path.exists("checkpoints/fewshot.pt") and os.path.exists("checkpoints/fewshot_words.pt"):
    for n, c in json.load(open(CATS, encoding="utf-8")).items():
        engine.add_category(n, **c)


class Category(BaseModel):
    name: str
    examples: list[str]
    not_examples: list[str] = []


class Sanitise(BaseModel):
    text: str
    session_id: Optional[str] = None


class Restore(BaseModel):
    text: str
    session_id: str


@app.post("/sanitise")
def sanitise(req: Sanitise):
    return engine.sanitise(req.text, req.session_id)


@app.post("/restore")
def restore(req: Restore):
    return {"restored_text": engine.restore(req.text, req.session_id)}


@app.post("/categories")
def add_category(req: Category):
    if not req.examples:
        raise HTTPException(422, "give at least one example")
    if not (os.path.exists("checkpoints/fewshot.pt") and os.path.exists("checkpoints/fewshot_words.pt")):
        raise HTTPException(503, "few-shot models missing: run python -m privylens.fewshot.run and python -m privylens.fewshot.words")
    label = engine.add_category(req.name, req.examples, req.not_examples)
    cats = json.load(open(CATS, encoding="utf-8")) if os.path.exists(CATS) else {}
    cats[label] = {"examples": req.examples, "not_examples": req.not_examples}
    json.dump(cats, open(CATS, "w", encoding="utf-8"))
    return {"label": label, "examples": len(req.examples)}


@app.get("/categories")
def list_categories():
    return json.load(open(CATS, encoding="utf-8")) if os.path.exists(CATS) else {}


@app.delete("/categories/{name}")
def delete_category(name: str):
    cats = json.load(open(CATS, encoding="utf-8")) if os.path.exists(CATS) else {}
    cats.pop(name, None)
    engine.custom.pop(name, None)
    json.dump(cats, open(CATS, "w", encoding="utf-8"))
    return {"deleted": name}


@app.post("/compare")
def compare(req: Sanitise):
    """The same text through three engines: regex only, regular NER (our tagger, no examples), PrivyLens (plus the user's examples)."""
    def one(**kw):
        spans = engine.detect(req.text, **kw)
        return {"sanitised_text": redact(req.text, [dict(s) for s in spans], {}), "spans": spans}
    out = {"regex": one(model=False, custom=False), "ner": one(custom=False), "privylens": one(), "distilbert": None}
    if os.path.exists(DISTILBERT):                           # the reference model is trained on Colab; absent = the page says so
        spans = [{**s, "conf": 1.0, "source": "model"} for s in _distilbert()(req.text)]
        out["distilbert"] = {"sanitised_text": redact(req.text, [dict(s) for s in spans], {}), "spans": spans}
    return out


DISTILBERT = "checkpoints/distilbert.pt"


@lru_cache(maxsize=1)
def _distilbert():
    """Loaded on first use (261 MB, slow to import), so the engine starts fast without it."""
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    from privylens.schema import ID2TAG, TAG2ID, TAGS
    from privylens.training.train_distilbert import NAME, predict
    hf = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForTokenClassification.from_pretrained(NAME, num_labels=len(TAGS), id2label=ID2TAG, label2id=TAG2ID)
    model.load_state_dict(torch.load(DISTILBERT, map_location="cpu"))
    return lambda text: predict(model, hf, text)
