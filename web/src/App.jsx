import { useEffect, useState } from "react";
import Compare from "./Compare.jsx";
import Docs from "./Docs.jsx";
import LiveView from "./LiveView.jsx";
import Teach from "./Teach.jsx";

const PAGES = { "#/compare": Compare, "#/live": LiveView, "#/docs": Docs, "#/teach": Teach };
const LINKS = [["#/compare", "Compare"], ["#/live", "Live view"], ["#/docs", "Docs"], ["#/teach", "Teach"]];

// Sentence for the lens demo: [text] is public, [text, label] is private.
const SAMPLE = [["Send the report to "], ["Rahul Sharma", "PERSON_1"], [" at "], ["rahul@example.com", "EMAIL_1"], [". His employee ID is "], ["ZX-48213-K", "ID_NUMBER_1"], ["."]];

// engine, size, % of private values left visible (reports/results.md: test, unseen templates)
const ROWS = [["Regex only", "no model", 68.7, 51.5], ["PrivyLens", "1.56M parameters", 2.7, 5.4], ["DistilBERT", "65.2M parameters", 2.3, 0.4]];

const STEPS = [
  ["Find", "Patterns catch emails, URLs, IP addresses and PAN. A small model catches names, places and organizations."],
  ["Merge", "Overlapping finds become one. When two engines disagree, the one that hides more wins."],
  ["Replace", "Each value becomes a placeholder like [PERSON_1]. The same value always gets the same one, and the originals stay in memory so replies can be restored."],
  ["Check", "The output is scanned again. If a structured value is still visible, the text is blocked instead of sent."],
];

function Lens() {
  const move = (e) => {
    const b = e.currentTarget.getBoundingClientRect();
    e.currentTarget.style.animation = "none";
    e.currentTarget.style.setProperty("--x", `${((e.clientX - b.left) / b.width) * 100}%`);
    e.currentTarget.style.setProperty("--y", `${((e.clientY - b.top) / b.height) * 100}%`);
  };
  const layer = (hide) => SAMPLE.map(([t, label], i) =>
    hide && label ? <span key={i} className="rounded bg-black text-transparent">{t}</span> : <span key={i}>{t}</span>);
  return (
    <div className="glass mt-14 overflow-hidden rounded-[2rem] p-8 sm:p-14">
      <div className="lens relative grid font-display text-3xl leading-[1.6] sm:text-5xl sm:leading-[1.6]" onPointerMove={move}>
        <p className="col-start-1 row-start-1">{layer(false)}</p>
        <p className="on pointer-events-none col-start-1 row-start-1" aria-hidden="true">{layer(true)}</p>
        <div className="ring" aria-hidden="true" />
      </div>
    </div>
  );
}

function Home() {
  return (
    <main className="mx-auto max-w-5xl px-4 pb-24 pt-36">
      <div className="glass rounded-[2rem] p-8 sm:p-14">
      <h1 className="font-display max-w-3xl text-5xl leading-[1.08] tracking-tight sm:text-7xl">Hide what is private before it leaves your computer.</h1>
      <p className="mt-6 max-w-xl text-lg">PrivyLens finds names, emails, ID numbers and your own kinds of secret in anything you are about to paste, and swaps them for placeholders. It runs on this machine. Nothing is uploaded.</p>
      <div className="mt-8 flex flex-wrap gap-3">
        <a href="#/compare" className="btn-ink">Compare the engines</a>
        <a href="#/docs" className="btn-glass glass">Read the docs</a>
      </div>
      </div>

      <Lens />
      <p className="glass mt-3 inline-block rounded-full px-4 py-1 text-sm">Move across the sentence. Under the glass, private values are hidden.</p>

      <section className="mt-28">
        <h2 className="glass font-display max-w-2xl rounded-3xl p-6 text-4xl leading-tight">Close to a model 42 times its size, and small enough to run in your browser.</h2>
        <div className="glass mt-8 overflow-x-auto rounded-3xl p-6 sm:p-8">
          <table className="w-full min-w-[34rem] text-left">
            <caption className="pb-5 text-left text-sm text-neutral-600">Private values still visible in the output (lower is better). Held-out test set and unseen sentence templates.</caption>
            <thead className="sr-only"><tr><th>Engine</th><th>Test set</th><th>Unseen templates</th></tr></thead>
            <tbody>
              {ROWS.map(([name, size, test, unseen]) => (
                <tr key={name} className="border-t border-black/15 align-top">
                  <th scope="row" className="py-4 pr-4 font-normal"><span className="block font-semibold">{name}</span><span className="text-sm text-neutral-600">{size}</span></th>
                  {[test, unseen].map((v, i) => (
                    <td key={i} className="w-1/3 py-4 pr-4">
                      <div className="h-3 rounded-full bg-black/10"><div className="h-3 rounded-full bg-black" style={{ width: `${v}%`, minWidth: 4 }} /></div>
                      <span className="mt-1 block font-mono text-sm">{v}% <span className="text-neutral-600">{i ? "unseen" : "test"}</span></span>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-4 text-sm text-neutral-600">DistilBERT leaks less on new sentence shapes. PrivyLens is the one that fits in a browser extension. <a className="underline" href="#/compare">Try your own text.</a></p>
        </div>
      </section>

      <section className="mt-28">
        <h2 className="glass font-display inline-block rounded-3xl px-6 py-3 text-4xl">What happens to your text</h2>
        <ol className="mt-8 grid gap-4 sm:grid-cols-2">
          {STEPS.map(([title, body], i) => (
            <li key={title} className="glass rounded-3xl p-6">
              <span className="font-mono text-sm text-neutral-600">{i + 1}</span>
              <h3 className="font-display mt-1 text-2xl">{title}</h3>
              <p className="mt-2">{body}</p>
            </li>
          ))}
        </ol>
      </section>
    </main>
  );
}

export default function App() {
  const [hash, setHash] = useState(location.hash);
  useEffect(() => {
    const f = () => { setHash(location.hash); scrollTo(0, 0); };
    addEventListener("hashchange", f);
    return () => removeEventListener("hashchange", f);
  }, []);
  const Page = PAGES[hash] || Home;
  return (
    <>
      <nav className="glass fixed left-1/2 top-4 z-20 flex w-[calc(100%-2rem)] max-w-2xl -translate-x-1/2 items-center justify-between rounded-full py-2 pl-5 pr-2" aria-label="Main">
        <a href="#/" className="font-display text-xl">PrivyLens</a>
        <div className="flex gap-1">
          {LINKS.map(([h, label]) => (
            <a key={h} href={h} aria-current={hash === h ? "page" : undefined}
              className={`rounded-full px-3 py-1.5 text-sm ${hash === h ? "bg-black text-white" : "hover:bg-black/10"}`}>{label}</a>
          ))}
        </div>
      </nav>
      <Page />
      <footer className="glass mx-auto mb-6 w-fit max-w-[calc(100%-2rem)] rounded-full px-5 py-2 text-sm">PrivyLens runs on your computer. Text you check here is never sent anywhere else.</footer>
    </>
  );
}
