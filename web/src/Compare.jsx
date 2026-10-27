import { useState } from "react";
import { call } from "./Teach.jsx";

const ENGINES = [
  ["regex", "Regex only", "Patterns for emails, phones, IDs. No model."],
  ["ner", "Regular NER", "Our small tagger on its own, with none of your examples."],
  ["distilbert", "DistilBERT", "A pretrained 65M-parameter model, fine-tuned on the same data."],
  ["privylens", "PrivyLens", "Tagger plus patterns plus the categories you taught it."],
];
const SAMPLES = [
  ["Everyday", "Send the report to Rahul Sharma at rahul@example.com. His employee ID is ZX-48213-K."],
  ["All caps", "MANAN MIHIR RAYCHAND DEDHIA, DOB 12/03/2001, lives in MUMBAI."],
  ["Documents", "Aadhaar 2345 6789 0123, PAN ABCDE1234F, call +91 98765 43210 or open https://intranet.acme.in/u/482."],
];

// Hidden values as black bars with their placeholder inside.
const show = (out) => out.sanitised_text.split(/(\[[A-Z_]+_\d+\])/).map((p, i) => (/^\[[A-Z_]+_\d+\]$/.test(p) ? <span key={i} className="bar">{p}</span> : p));

// What PrivyLens hides that this engine left visible.
const missed = (text, mine, ref) => ref.spans.filter((r) => !mine.spans.some((s) => s.start < r.end && s.end > r.start)).map((r) => text.slice(r.start, r.end));

export default function Compare() {
  const [text, setText] = useState(SAMPLES[0][1]);
  const [res, setRes] = useState(null);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const run = () => {
    setErr(""); setBusy(true);
    call("/compare", "POST", { text }).then((r) => setRes({ ...r, text })).catch((e) => { setRes(null); setErr(e.message); }).finally(() => setBusy(false));
  };
  return (
    <main className="mx-auto max-w-5xl px-4 pb-24 pt-32">
      <div className="glass rounded-[2rem] p-8">
        <h1 className="font-display text-5xl leading-tight">Same text, four engines.</h1>
        <p className="mt-3 max-w-xl text-lg">Paste something with private details in it. Each engine redacts it independently, so you can see who hides what.</p>
      </div>

      <div className="glass mt-8 rounded-3xl p-5">
        <label htmlFor="t" className="sr-only">Text to redact</label>
        <textarea id="t" rows={4} value={text} onChange={(e) => setText(e.target.value)}
          className="w-full rounded-2xl border border-black/20 bg-white/70 p-3 font-mono text-sm" />
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <button onClick={run} disabled={busy || !text.trim()} className="btn-ink disabled:opacity-40">{busy ? "Comparing..." : "Compare"}</button>
          <span className="ml-2 text-sm text-neutral-600">Try:</span>
          {SAMPLES.map(([n, t]) => <button key={n} onClick={() => setText(t)} className="rounded-full border border-black/25 bg-white/50 px-3 py-1 text-sm hover:bg-white">{n}</button>)}
        </div>
      </div>

      {err && <p role="alert" className="glass mt-6 rounded-2xl p-4">Could not compare: {err === "Failed to fetch" ? "the engine is not running. Start it with python -m uvicorn api.server:app --port 8000" : err}</p>}

      <div className="mt-6 grid gap-4 md:grid-cols-2">
        {ENGINES.map(([key, name, blurb]) => {
          const out = res && res[key];
          return (
            <section key={key} className={`glass rounded-3xl p-6 ${key === "privylens" ? "ring-2 ring-black" : ""}`} aria-labelledby={key}>
              <h2 id={key} className="font-display text-2xl">{name}</h2>
              <p className="text-sm text-neutral-600">{blurb}</p>
              <div className="mt-4 min-h-24 leading-9" data-testid={key}>
                {!res ? <span className="text-neutral-500">Press Compare to see the output.</span>
                  : out ? show(out)
                  : <span>Not loaded. Fine-tune it on Colab (<code>python -m privylens.training.train_distilbert</code>), save it as <code>checkpoints/distilbert.pt</code> and restart the engine.</span>}
              </div>
              {out && (
                <p className="mt-3 text-sm text-neutral-700">
                  {out.spans.length} hidden.
                  {key !== "privylens" && (() => { const m = missed(res.text, out, res.privylens); return m.length ? <> Left visible that PrivyLens hides: <b>{m.join(", ")}</b></> : " Nothing PrivyLens hides was left visible."; })()}
                </p>
              )}
            </section>
          );
        })}
      </div>
      <p className="glass mt-4 inline-block rounded-full px-4 py-1 text-sm">Taught PrivyLens a category of your own? Only the last card uses it. <a className="underline" href="#/teach">Teach one</a>.</p>
    </main>
  );
}
