import { useEffect, useState } from "react";
import T from "./trace.json";   // every number below comes from the real model: python -m privylens.evaluation.trace

const softmax = (a) => { const m = Math.max(...a), e = a.map((x) => Math.exp(x - m)), s = e.reduce((p, c) => p + c); return e.map((x) => x / s); };
const f = (v, n = 2) => v.toFixed(n);
const pct = (v) => `${(v * 100).toFixed(v < 0.1 ? 1 : 0)}%`;
const shade = (v) => `hsl(0 0% ${50 - Math.max(-1, Math.min(1, v / 3)) * 46}%)`;       // dark = positive, light = negative

const STAGES = [
  { k: "split", title: "Cut into tokens", text: "The model cannot read letters or words. A tokenizer cuts the text into pieces it has seen often, such as R, ah and ul for Rahul. Every piece has an id in a vocabulary of 8,000." },
  { k: "embed", title: "Look up numbers", text: "Each id picks one row from a table of 8,000 rows. A row is 128 numbers describing the piece. Only the first 8 are shown. They start random and are tuned during training." },
  { k: "pos", title: "Add the position", text: "Attention has no sense of order, so each token gets a pattern of sine and cosine waves for its place in the sentence, added number by number." },
  ...T.layers.flatMap((_, l) => [
    { k: "attn", l, title: `Layer ${l + 1}: attention`, text: "Every token looks at every other token and borrows from the ones that matter. Each of the 4 heads looks in its own way. The result is added to the input and the sum is rescaled (LayerNorm)." },
    { k: "ffn", l, title: `Layer ${l + 1}: feed-forward`, text: "A small network works on each token on its own: 128 numbers widen to 256, pass through GELU, shrink back to 128, and are added to what went in." },
  ]),
  { k: "score", title: "Score every tag", text: "A final layer turns each token's 128 numbers into one score per tag (23 tags). Softmax turns the scores into percentages that add up to 100." },
  { k: "spans", title: "Group tags into spans", text: "B- starts a hidden value, I- continues it and O means ordinary text. Neighbouring pieces with the same label become one span." },
  { k: "merge", title: "Merge with patterns", text: "Pattern matching runs on the same text. Precise patterns (email, URL, IP, PAN) outrank the model, and the model outranks loose patterns. Overlaps keep one winner." },
  { k: "redact", title: "Redact", text: "Each winning span is replaced by a numbered placeholder. The same value always gets the same placeholder, and the originals stay in memory so replies can be restored." },
];
const last = STAGES.length - 1;

const Nums = ({ label, v, note }) => (
  <div className="mt-2">
    <div className="text-xs text-neutral-600">{label}</div>
    <div className="mt-0.5 flex flex-wrap items-center gap-1">
      {v.map((x, i) => <span key={i} className="w-10 rounded bg-white/80 px-0.5 text-right font-mono text-[11px]" style={{ borderBottom: `3px solid ${shade(x)}` }}>{f(x)}</span>)}
      {note && <span className="ml-1 font-mono text-xs text-neutral-600">{note}</span>}
    </div>
  </div>
);

function Inspector({ s, n, head, setHead }) {
  const tok = T.tokens[n], word = T.words[tok.word];
  const l = T.layers[s.l];
  if (s.k === "split") {
    const mine = T.tokens.filter((t) => t.word === tok.word);
    return <p>The word <b>{word.text}</b> becomes {mine.length} token{mine.length > 1 ? "s" : ""}: {mine.map((t, i) => <span key={i} className="mr-1 rounded bg-black px-1.5 font-mono text-xs text-white">{t.text} #{t.id}</span>)}</p>;
  }
  if (s.k === "embed") return <>
    <p>Token <b>{tok.text}</b>, id {tok.id}: row {tok.id} of the table, times √128 = {f(Math.sqrt(T.d_model))}.</p>
    <Nums label="First 8 of 128 numbers" v={tok.emb} />
  </>;
  if (s.k === "pos") return <>
    <p>Position {n} of {T.tokens.length}. Wave value = sin or cos of {n} / 10000<sup>2i/128</sup>.</p>
    <Nums label="Embedding" v={tok.emb} /><Nums label="+ Position" v={tok.pe} /><Nums label="= Sum, the input to layer 1" v={tok.x0} />
  </>;
  if (s.k === "attn") {
    const sc = l.scores[head][n], w = softmax(sc);
    const top = w.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, 6);
    const prev = s.l ? T.layers[s.l - 1].out[n] : tok.x0;
    return <>
      <div role="group" aria-label="Attention head" className="flex gap-1">
        {l.scores.map((_, h) => <button key={h} onClick={() => setHead(h)} aria-pressed={head === h} className={`rounded-full px-3 py-1 text-xs ${head === h ? "bg-black text-white" : "bg-white/70"}`}>Head {h + 1}</button>)}
      </div>
      <p className="mt-3 text-sm"><b>{tok.text}</b> asks every token a question. Score = (query · key) / √32, then softmax:</p>
      <ul className="mt-2 space-y-1">
        {top.map(([p, i]) => (
          <li key={i} className="grid grid-cols-[5rem_1fr_6.5rem] items-center gap-2 text-xs">
            <span className="truncate font-mono">{T.tokens[i].text}</span>
            <span className="h-2 rounded-full bg-black/10"><span className="block h-2 rounded-full bg-black" style={{ width: pct(p) }} /></span>
            <span className="text-right font-mono">{f(sc[i])} → {pct(p)}</span>
          </li>))}
      </ul>
      <p className="mt-1 text-xs text-neutral-600">The other {w.length - top.length} tokens share {pct(1 - top.reduce((a, [p]) => a + p, 0))}.</p>
      <Nums label="Input" v={prev} /><Nums label="+ What attention added (all 4 heads)" v={l.attn_out[n]} /><Nums label="= Sum" v={l.resid1[n]} />
      <Nums label="LayerNorm: (x - mean) / std" v={l.ln1[n]} note={`mean ${f(l.ln1_stats[n][0])}, std ${f(l.ln1_stats[n][1])}`} />
    </>;
  }
  if (s.k === "ffn") return <>
    <Nums label={`Widened to ${T.d_ff}, after GELU (first 8)`} v={l.hidden[n]} />
    <Nums label="Shrunk to 128, plus the input" v={l.resid2[n]} />
    <Nums label="LayerNorm: layer output" v={l.out[n]} note={`mean ${f(l.out_stats[n][0])}, std ${f(l.out_stats[n][1])}`} />
  </>;
  if (s.k === "score") {
    const p = softmax(tok.logits);
    const top = p.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 4);
    return <>
      <p className="text-sm">Scores for <b>{tok.text}</b>, top 4 of {T.tags.length}. Softmax: e<sup>score</sup> / sum of all e<sup>score</sup>.</p>
      <ul className="mt-2 space-y-1">
        {top.map(([v, i]) => (
          <li key={i} className="grid grid-cols-[7.5rem_1fr_6.5rem] items-center gap-2 text-xs">
            <span className="truncate font-mono">{T.tags[i]}</span>
            <span className="h-2 rounded-full bg-black/10"><span className="block h-2 rounded-full bg-black" style={{ width: pct(v) }} /></span>
            <span className="text-right font-mono">{f(tok.logits[i])} → {pct(v)}</span>
          </li>))}
      </ul>
    </>;
  }
  if (s.k === "spans") return (
    <ul className="space-y-1 text-sm">
      {T.spans.map((x, i) => <li key={i}><span className="rounded bg-black px-1.5 font-mono text-xs text-white">{x.label}</span> {T.text.slice(x.start, x.end)}</li>)}
    </ul>);
  if (s.k === "merge") return <>
    <p className="text-sm">Patterns found: {T.regex.length ? T.regex.map((x) => `${x.label} "${T.text.slice(x.start, x.end)}"`).join(", ") : "nothing"}.</p>
    <p className="mt-1 text-sm">The model found {T.model.length} spans, including the same email. Winners:</p>
    <ul className="mt-2 space-y-1 text-sm">
      {T.final.map((x, i) => <li key={i}><span className="rounded bg-black px-1.5 font-mono text-xs text-white">{x.label}</span> {T.text.slice(x.start, x.end)} <span className="text-xs text-neutral-600">from {x.source === "regex" ? "a pattern" : "the model"}</span></li>)}
    </ul>
  </>;
  return <>
    <p className="leading-9">{T.redacted.split(/(\[[A-Z_]+_\d+\])/).map((p, i) => (/^\[/.test(p) ? <span key={i} className="bar">{p}</span> : p))}</p>
    <ul className="mt-3 space-y-1 font-mono text-xs">{Object.entries(T.vault).map(([v, ph]) => <li key={ph}>{ph} = {v} <span className="text-neutral-600">(kept in memory only)</span></li>)}</ul>
  </>;
}

// What a row shows to the right of its strip at this stage.
function note(s, n, head) {
  const tok = T.tokens[n], span = T.final.find((x) => tok.start >= x.start && tok.end <= x.end);
  if (s.k === "split") return `#${tok.id}`;
  if (s.k === "embed") return `row ${tok.id}`;
  if (s.k === "pos") return `pos ${n}`;
  if (s.k === "attn") { const sc = T.layers[s.l].scores[head][n]; return `looks at ${T.tokens[sc.indexOf(Math.max(...sc))].text}`; }
  if (s.k === "score") return `${tok.tag} ${pct(tok.p)}`;
  if (s.k === "spans") return tok.tag;
  if (s.k === "merge") return span ? `${span.label} (${span.source === "regex" ? "pattern" : "model"})` : "";
  if (s.k === "redact" && span) return T.tokens.find((x) => x.start >= span.start && x.end <= span.end) === tok ? T.vault[T.text.slice(span.start, span.end)] : "";
  return "";
}

export default function LiveView() {
  const [stage, setStage] = useState(0);
  const [sel, setSel] = useState(4);
  const [head, setHead] = useState(0);
  const [playing, setPlaying] = useState(() => !matchMedia("(prefers-reduced-motion: reduce)").matches);
  useEffect(() => {
    if (!playing) return;
    if (stage >= last) { setPlaying(false); return; }
    const t = setTimeout(() => setStage(stage + 1), 3500);
    return () => clearTimeout(t);
  }, [playing, stage]);
  const go = (s) => { setPlaying(false); setStage(Math.max(0, Math.min(last, s))); };
  const s = STAGES[stage];
  const vecOf = (n) => ({ embed: T.tokens[n].emb, pos: T.tokens[n].x0, attn: T.layers[s.l]?.ln1[n], ffn: T.layers[s.l]?.out[n] })[s.k];

  return (
    <main className="mx-auto max-w-6xl px-4 pb-24 pt-32">
      <div className="glass rounded-[2rem] p-8">
        <h1 className="font-display text-5xl leading-tight">Live view</h1>
        <p className="mt-3 max-w-2xl text-lg">One sentence goes through the real PrivyLens model. Watch each token change, step by step, with the actual numbers. Click a token to follow it.</p>
      </div>

      <div className="glass mt-4 flex flex-wrap items-center gap-3 rounded-3xl p-4">
        <button onClick={() => go(stage - 1)} disabled={stage === 0} className="btn-glass glass disabled:opacity-40">Back</button>
        <button onClick={() => (stage >= last ? (setStage(0), setPlaying(true)) : setPlaying(!playing))} className="btn-ink">{stage >= last && !playing ? "Replay" : playing ? "Pause" : "Play"}</button>
        <button onClick={() => go(stage + 1)} disabled={stage === last} className="btn-glass glass disabled:opacity-40">Next</button>
        <label className="flex min-w-48 flex-1 items-center gap-3 text-sm">
          <span className="sr-only">Step</span>
          <input type="range" min={0} max={last} value={stage} onChange={(e) => go(+e.target.value)} className="w-full accent-black" />
        </label>
        <span className="font-mono text-sm">{stage + 1} / {STAGES.length}</span>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-[1fr_26rem]">
        <section className="glass rounded-3xl p-4 sm:p-6" aria-label="Tokens">
          <h2 className="font-display text-2xl">{s.title}</h2>
          <p className="mt-1 max-w-xl text-sm">{s.text}</p>
          <ul className="mt-4">
            {T.tokens.map((t, n) => {
              const v = vecOf(n), span = T.final.find((x) => t.start >= x.start && t.end <= x.end);
              const hidden = s.k === "redact" && span, first = n === 0 || T.tokens[n - 1].word !== t.word;
              return (
                <li key={n}>
                  <button onClick={() => setSel(n)} aria-pressed={sel === n}
                    className={`grid w-full grid-cols-[4.5rem_5.5rem_1fr] items-center gap-2 rounded-xl px-2 py-1 text-left sm:grid-cols-[6rem_6rem_1fr_9rem] ${sel === n ? "bg-black/10 ring-1 ring-black" : "hover:bg-black/5"}`}>
                    <span className="truncate text-xs text-neutral-600">{first ? T.words[t.word].text : ""}</span>
                    <span className={`truncate font-mono text-sm ${hidden ? "rounded bg-black text-transparent" : ""}`}>{t.text}</span>
                    <span className="flex gap-0.5" aria-hidden="true">
                      {v ? v.map((x, i) => <span key={i} className="h-4 w-4 rounded-sm motion-safe:transition-colors motion-safe:duration-700 sm:w-5" style={{ background: shade(x) }} />) : <span className="h-4" />}
                    </span>
                    <span className="hidden truncate font-mono text-xs sm:block">{note(s, n, head)}</span>
                  </button>
                </li>);
            })}
          </ul>
          <p className="mt-3 text-xs text-neutral-600">Each square is one of the token's numbers (first 8 of 128). Dark is positive, light is negative.</p>
        </section>

        <aside className="glass glass-solid h-fit rounded-3xl p-5 lg:sticky lg:top-24" aria-live="polite">
          <h2 className="font-display text-2xl">Token <span className="rounded bg-black px-2 font-mono text-lg text-white">{T.tokens[sel].text}</span></h2>
          <p className="mb-3 text-xs text-neutral-600">in the word "{T.words[T.tokens[sel].word].text}", position {sel}</p>
          <Inspector s={s} n={sel} head={head} setHead={setHead} />
        </aside>
      </div>
      <p className="glass mt-4 inline-block rounded-full px-4 py-1 text-sm">Sentence: {T.text}</p>
    </main>
  );
}
