const Code = ({ children }) => <pre className="mt-3 overflow-x-auto rounded-2xl bg-black p-4 font-mono text-sm leading-relaxed text-white">{children}</pre>;

const SECTIONS = [
  ["start", "Start the engine", <>
    <p>The engine is a small local server. It binds to 127.0.0.1, so only this computer can reach it.</p>
    <Code>{`pip install -r requirements.txt\npython -m uvicorn api.server:app --host 127.0.0.1 --port 8000`}</Code>
    <p className="mt-3">It uses <code>checkpoints/tagger.pt</code> when that file exists. Without it, only pattern matching runs.</p>
  </>],
  ["web", "Run this website", <>
    <Code>{`cd web\nnpm install\nnpm run dev`}</Code>
    <p className="mt-3">Open the address Vite prints. The Compare and Teach pages talk to the engine on port 8000.</p>
  </>],
  ["extension", "Install the browser extension", <>
    <p>Works in Chrome, Edge and Brave. PrivyLens is not in the Chrome Web Store, so Chrome asks you to load it by hand. It takes a minute.</p>
    <a href="privylens-extension.zip" download className="btn-ink mt-4 inline-block">Download the extension (zip)</a>
    <ol className="mt-5 list-decimal space-y-2 pl-5">
      <li>Unzip the file into a folder you will keep. Chrome reads the extension from there, so do not delete it afterwards.</li>
      <li>Open <code>chrome://extensions</code> in a new tab (Edge: <code>edge://extensions</code>).</li>
      <li>Turn on <b>Developer mode</b> in the top right.</li>
      <li>Click <b>Load unpacked</b> and choose the unzipped folder.</li>
      <li>Click the puzzle-piece icon in the toolbar and pin PrivyLens.</li>
    </ol>
    <p className="mt-3">The extension needs the engine running on your computer. See Start the engine above. If the popup says it cannot reach the engine, start it and try again.</p>
  </>],
  ["use", "How to use it", <>
    <ol className="list-decimal space-y-3 pl-5">
      <li><b>Start the engine</b> and check the PrivyLens popup shows no error.</li>
      <li><b>Paste as usual</b> into ChatGPT, Claude, Gemini, Copilot or Perplexity. With "Clean pastes automatically" on, private values become placeholders like <code>[PERSON_1]</code> before the text is sent.</li>
      <li><b>Any other site:</b> click in a text box and press <b>Alt+Shift+P</b>. It cleans your selection, or the whole box if nothing is selected. To make a site clean automatically, open the popup on it and add the site.</li>
      <li><b>Check before inserting:</b> turn on "Ask me before inserting" to see a preview. Click a highlight to keep that value visible. Dashed highlights are ones the model was unsure about.</li>
      <li><b>Get the real values back:</b> paste the AI's reply into the Restore box in the popup. <code>[PERSON_1]</code> turns back into the name. Press New session to start numbering again.</li>
      <li><b>Manual mode:</b> paste text into the popup, press Sanitise and copy the result.</li>
    </ol>
    <p className="mt-3">Your own kinds of secret, such as staff codes, can be added in the popup or on the Teach page.</p>
  </>],
  ["api", "API", <>
    <p><code>POST /sanitise</code> takes <code>{"{text, session_id?}"}</code> and returns the redacted text, the spans, a <code>blocked</code> flag and the session id.</p>
    <Code>{`curl -s localhost:8000/sanitise -H "content-type: application/json" \\\n  -d '{"text":"Mail rahul@example.com"}'\n# {"sanitised_text":"Mail [EMAIL_1]", "blocked":false, ...}`}</Code>
    <p className="mt-3"><code>POST /restore</code> takes <code>{"{text, session_id}"}</code> and puts the originals back, for example into a chatbot's reply.</p>
    <p className="mt-3"><code>POST /compare</code> takes <code>{"{text}"}</code> and returns <code>regex</code>, <code>ner</code>, <code>privylens</code> and <code>distilbert</code>. The last is <code>null</code> until the DistilBERT checkpoint exists.</p>
    <p className="mt-3"><code>GET /categories</code>, <code>POST /categories</code> and <code>DELETE /categories/{"{name}"}</code> manage the categories you taught.</p>
  </>],
  ["teach", "Teach a new kind of secret", <>
    <p>Give a name and a few examples. Nothing is retrained. The best example is a sentence with the secret in double brackets:</p>
    <Code>{`Staff code: [[ORX-72841]]\n[[Nightfall]] is the internal codename for the launch`}</Code>
    <p className="mt-3">Values containing a digit go to the ID model. Capitalized names need a bracketed sentence. Every example is also matched literally. Add things that look similar but are not secret, and they are left alone. This needs <code>checkpoints/fewshot.pt</code> and <code>fewshot_words.pt</code>.</p>
  </>],
  ["placeholders", "Placeholders and restore", <p>A hidden value becomes <code>[LABEL_n]</code>. The same value always gets the same placeholder inside a session, so a conversation stays coherent. The mapping lives in the engine's memory only and is gone when it stops.</p>],
  ["compare", "Add DistilBERT to the comparison", <>
    <p>DistilBERT is a reference model, not part of the product. Fine-tune it on the same data, ideally on a Colab GPU, then copy the file next to the others.</p>
    <Code>{`python -m privylens.training.train_distilbert --epochs 4\n# saves checkpoints/distilbert.pt; restart the engine`}</Code>
  </>],
  ["limits", "What it gets wrong", <ul className="list-disc space-y-2 pl-5">
    <li>Lowercase place names can be missed, and that leaves them visible.</li>
    <li>Some common words are hidden by mistake, such as "His" or "Mail me at". That is over-redaction, not a leak.</li>
    <li>Title Case headings in PDFs can be taken for names. The PDF tool has <code>--keep</code> and <code>--hide</code> for that.</li>
    <li>Scanned PDFs have no text layer and are not handled yet.</li>
    <li>About 2.7% of private values stay visible on the held-out test set, and 5.4% on unseen sentence shapes. Check anything sensitive before you send it.</li>
  </ul>],
];

export default function Docs() {
  const go = (id) => document.getElementById(id)?.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  return (
    <main className="mx-auto grid max-w-5xl gap-8 px-4 pb-24 pt-32 md:grid-cols-[13rem_1fr]">
      <nav className="glass h-fit rounded-3xl p-3 md:sticky md:top-24" aria-label="Docs">
        <ul className="flex flex-wrap gap-1 md:block">
          {SECTIONS.map(([id, title]) => <li key={id}><button onClick={() => go(id)} className="w-full rounded-full px-3 py-1.5 text-left text-sm hover:bg-black/10">{title}</button></li>)}
        </ul>
      </nav>
      <div className="min-w-0">
        <div className="glass rounded-[2rem] p-8">
          <h1 className="font-display text-5xl leading-tight">Docs</h1>
          <p className="mt-3 max-w-xl text-lg">Everything runs on your computer. These are the pieces and how to start them.</p>
        </div>
        <div className="mt-8 space-y-4">
          {SECTIONS.map(([id, title, body]) => (
            <section key={id} id={id} className="glass glass-solid scroll-mt-24 rounded-3xl p-6 [&_code]:rounded [&_code]:bg-black/10 [&_code]:px-1 [&_code]:font-mono [&_code]:text-[.9em] [&_pre_code]:bg-transparent [&_pre_code]:p-0">
              <h2 className="font-display mb-3 text-2xl">{title}</h2>
              {body}
            </section>
          ))}
        </div>
      </div>
    </main>
  );
}
