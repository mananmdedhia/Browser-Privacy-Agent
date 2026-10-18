"""Local-only HTTP wrapper around the privacy engine, for the browser extension.

Run:  python -m uvicorn api.server:app --host 127.0.0.1 --port 8000
      (127.0.0.1 = this computer only; text never leaves the machine)
POST /sanitise  {text, session_id?}        -> {sanitised_text, spans, flagged, blocked, leaks, session_id}
POST /restore   {text, session_id}         -> {restored_text}
Uses the trained tagger if checkpoints/tagger.pt exists, otherwise regex only.
"""
import os
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from privylens.engine.pipeline import Engine

app = FastAPI(title="PrivyLens local engine")
app.add_middleware(CORSMiddleware, allow_origin_regex=r"chrome-extension://.*|http://(localhost|127\.0\.0\.1)(:\d+)?",
                   allow_methods=["POST"], allow_headers=["*"])
engine = Engine.from_checkpoint() if os.path.exists("checkpoints/tagger.pt") else Engine()


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
