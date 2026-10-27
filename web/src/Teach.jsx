import { useEffect, useState } from "react";

export const API = "http://127.0.0.1:8000";
export const call = (path, method = "GET", body) =>
  fetch(API + path, { method, headers: { "content-type": "application/json" }, body: body && JSON.stringify(body) })
    .then(async (r) => { if (!r.ok) throw new Error((await r.json()).detail || r.statusText); return r.json(); });
const lines = (s) => s.split("\n").map((x) => x.trim()).filter(Boolean);

// One hidden value: a black bar that shows the original text on click.
function Bar({ label, original }) {
  const [open, setOpen] = useState(false);
  return (
    <button onClick={() => setOpen(!open)} title={label} aria-pressed={open}
      className={`px-1 transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 ${open ? "bg-white text-black ring-1 ring-black" : "bg-black text-black"}`}>
      {original}
    </button>
  );
}

function Redactor({ onClose }) {
  const [text, setText] = useState("Send the report to Rahul Sharma at rahul@example.com. His employee ID is ZX-48213-K.");
  const [res, setRes] = useState(null);
  const [err, setErr] = useState("");
  const run = () => {
    setErr("");
    call("/sanitise", "POST", { text }).then((r) => setRes({ ...r, source: text })).catch((e) => setErr(e.message));
  };
  // split the sanitised text on its placeholders and pair each with the original it hides
  const parts = res && res.sanitised_text.split(/(\[[A-Z_]+_\d+\])/).map((p, i) => {
    const s = res.spans.find((x) => x.placeholder === p);
    return s ? <Bar key={i} label={s.label} original={res.source.slice(s.start, s.end)} /> : p;
  });
  return (
    <div className="fixed inset-0 z-10 flex items-center justify-center bg-black/60 p-4" onClick={onClose}>
      <div role="dialog" aria-modal="true" aria-label="Redact text" onClick={(e) => e.stopPropagation()}
        className="glass glass-solid max-h-full w-full max-w-2xl overflow-auto rounded-3xl p-6 text-black">
        <div className="flex justify-between">
          <h2 className="text-xl">Redact text</h2>
          <button onClick={onClose} className="underline">Close</button>
        </div>
        <textarea value={text} onChange={(e) => setText(e.target.value)} rows={4}
          className="mt-4 w-full border border-black p-2 font-mono text-sm" />
        <button onClick={run} className="btn-ink mt-3">Redact</button>
        {err && <p className="mt-3">Could not redact: {err}. Is the engine running?</p>}
        {res && (
          <>
            <p className="mt-6 leading-8" data-testid="out">{parts}</p>
            <p className="mt-3 text-sm text-neutral-500">
              {res.spans.length} hidden. Click a bar to see what it covers.
              {res.spans.some((s) => s.source === "custom") && " Black bars with your own labels came from categories you taught."}
            </p>
          </>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [cats, setCats] = useState({});
  const [name, setName] = useState("");
  const [ex, setEx] = useState("");
  const [no, setNo] = useState("");
  const [msg, setMsg] = useState("");
  const [modal, setModal] = useState(false);
  const refresh = () => call("/categories").then(setCats).catch(() => setMsg("Engine not reachable. Start it: python -m uvicorn api.server:app --port 8000"));
  useEffect(() => { refresh(); }, []);

  const teach = (e) => {
    e.preventDefault();
    setMsg("");
    call("/categories", "POST", { name, examples: lines(ex), not_examples: lines(no) })
      .then((r) => { setMsg(`Learned ${r.label} from ${r.examples} example${r.examples > 1 ? "s" : ""}.`); setName(""); setEx(""); setNo(""); refresh(); })
      .catch((e) => setMsg(`Could not add it: ${e.message}`));
  };
  const remove = (n) => call(`/categories/${n}`, "DELETE").then(refresh);

  return (
    <>
      <main className="mx-auto max-w-3xl px-4 pb-24 pt-32">
        <div className="glass rounded-[2rem] p-8">
        <h1 className="font-display text-4xl leading-tight sm:text-5xl">Show it a few examples. It hides the rest.</h1>
        <p className="mt-4 max-w-xl text-lg">Every company has its own kinds of secret: staff codes, ticket keys, account numbers. Add one example of yours and PrivyLens starts hiding that kind of value straight away. Nothing is retrained and nothing leaves this computer.</p>
        <button onClick={() => setModal(true)} className="btn-ink mt-6">Try it on some text</button>
        </div>

        <h2 className="glass font-display mt-12 inline-block rounded-3xl px-5 py-2 text-2xl">Teach a new kind of secret</h2>
        <form onSubmit={teach} className="glass mt-4 space-y-4 rounded-3xl p-6">
          <label className="block">Name
            <input required value={name} onChange={(e) => setName(e.target.value)} placeholder="Zorvex employee ID"
              className="mt-1 block w-full rounded-xl border border-black/30 bg-white/60 p-2" />
          </label>
          <label className="block">Examples, one per line
            <textarea required rows={3} value={ex} onChange={(e) => setEx(e.target.value)} placeholder={"Employee ZX-11502-A joined today\nZX-90017-B"}
              className="mt-1 block w-full rounded-xl border border-black/30 bg-white/60 p-2 font-mono text-sm" />
          </label>
          <label className="block">Things that look similar but are not secret (optional)
            <textarea rows={2} value={no} onChange={(e) => setNo(e.target.value)} placeholder="Order number 48213"
              className="mt-1 block w-full rounded-xl border border-black/30 bg-white/60 p-2 font-mono text-sm" />
          </label>
          <button className="btn-ink">Add category</button>
          {msg && <p role="status">{msg}</p>}
        </form>

        <h2 className="glass font-display mt-12 inline-block rounded-3xl px-5 py-2 text-2xl">What it hides now</h2>
        <p className="glass mt-3 rounded-2xl p-4">Built in: names, places, organizations, emails, phones, dates, Aadhaar, PAN, URLs, IP addresses, ID numbers.</p>
        {Object.keys(cats).length === 0
          ? <p className="glass mt-4 rounded-2xl p-4">No categories of your own yet. Add one above.</p>
          : <ul className="mt-4 divide-y divide-current border-y border-current">
              {Object.entries(cats).map(([n, c]) => (
                <li key={n} className="flex items-start justify-between gap-4 py-3">
                  <div>
                    <div className="font-mono">[{n}_1]</div>
                    <div className="text-sm text-neutral-500">{c.examples.length} example{c.examples.length > 1 ? "s" : ""}: {c.examples.join("; ")}</div>
                  </div>
                  <button onClick={() => remove(n)} className="underline">Remove</button>
                </li>
              ))}
            </ul>}
        <p className="glass mt-10 rounded-2xl p-4 text-sm">To use PrivyLens while you browse, load the <code>extension</code> folder in Chrome (chrome://extensions, Developer mode, Load unpacked).</p>
      </main>
      {modal && <Redactor onClose={() => setModal(false)} />}
    </>
  );
}
