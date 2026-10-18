// PrivyLens popup. Talks only to the engine on THIS computer (127.0.0.1); nothing goes to the internet.
//
// Use it:  1) start the engine:  python -m uvicorn api.server:app --host 127.0.0.1 --port 8000
//          2) chrome://extensions -> turn on Developer mode -> Load unpacked -> pick this `extension` folder
//          3) select text on any page, click the PrivyLens icon, press Sanitise.
const API = "http://127.0.0.1:8000";
const $ = (id) => document.getElementById(id);
let state = null;                       // {chars, spans, blocked} for the text currently under review

// The engine counts characters like Python (code points); JS strings count UTF-16 units.
// Array.from(text) splits by code points, so offsets from the server line up even with emoji.
const slice = (chars, a, b) => chars.slice(a, b).join("");

async function post(path, body) {
  const r = await fetch(API + path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error("engine returned " + r.status);
  return r.json();
}

function showError(msg) { $("error").textContent = msg; $("error").hidden = !msg; }

async function pageSelection() {        // whatever the user highlighted on the page (also inside text boxes)
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const [res] = await chrome.scripting.executeScript({ target: { tabId: tab.id }, func: () => {
      const el = document.activeElement;
      if (el && typeof el.selectionStart === "number" && el.selectionStart !== el.selectionEnd)
        return el.value.slice(el.selectionStart, el.selectionEnd);
      return String(getSelection());
    } });
    return res.result || "";
  } catch { return ""; }                // not running as an extension, or a page Chrome does not allow reading
}

function hue(label) {                   // same label -> same colour
  let h = 0;
  for (const c of label) h = (h * 31 + c.charCodeAt(0)) % 360;
  return h;
}

function render() {
  const { chars, spans, blocked } = state;
  const marked = $("marked"), out = [];
  marked.textContent = "";
  let pos = 0;
  for (const s of spans) {
    marked.append(slice(chars, pos, s.start));
    const el = document.createElement("span");          // textContent only: page text is untrusted, never innerHTML
    el.className = "hl" + (s.flagged ? " flagged" : "") + (s.on ? "" : " off");
    el.style.setProperty("--h", hue(s.label));
    el.title = `${s.label} -> ${s.placeholder} (confidence ${s.conf.toFixed(2)}, found by ${s.source}). Click to ${s.on ? "keep visible" : "hide again"}`;
    el.append(slice(chars, s.start, s.end));
    const tag = document.createElement("small");
    tag.textContent = s.label;
    el.append(tag);
    el.onclick = () => { s.on = !s.on; render(); };
    marked.append(el);
    out.push(slice(chars, pos, s.start), s.on ? s.placeholder : slice(chars, s.start, s.end));
    pos = s.end;
  }
  marked.append(slice(chars, pos));
  out.push(slice(chars, pos));
  $("output").value = out.join("");

  const notes = [];
  if (blocked) notes.push("BLOCKED: structured private data is still visible in the output. Do not share it.");
  const unsure = spans.filter((s) => s.flagged && s.on).length;
  if (unsure) notes.push(`${unsure} low-confidence highlight(s) (dashed): check them.`);
  if (!spans.length) notes.push("Nothing private was detected. Double-check before sharing.");
  $("notes").textContent = notes.join(" ");
  $("notes").className = "banner " + (blocked ? "error" : "warn");
  $("notes").hidden = !notes.length;
  $("copy").disabled = blocked;
  $("review").hidden = false;
}

$("go").onclick = async () => {
  const text = $("input").value;
  if (!text.trim()) return;
  showError("");
  try {
    const r = await post("/sanitise", { text, session_id: localStorage.getItem("session_id") || null });
    localStorage.setItem("session_id", r.session_id);   // same session = same value keeps the same placeholder
    const flagged = new Set(r.flagged.map((s) => s.start + ":" + s.end));
    state = { chars: Array.from(text), blocked: r.blocked,
              spans: r.spans.map((s) => ({ ...s, on: true, flagged: flagged.has(s.start + ":" + s.end) })) };
    render();
  } catch (e) {
    showError("Cannot reach the PrivyLens engine. Start it with: python -m uvicorn api.server:app --host 127.0.0.1 --port 8000");
  }
};

$("newSession").onclick = () => { localStorage.removeItem("session_id"); $("review").hidden = true; showError(""); };

$("copy").onclick = async () => {
  await navigator.clipboard.writeText($("output").value);
  $("copy").textContent = "Copied!";
  setTimeout(() => ($("copy").textContent = "Copy sanitised text"), 1200);
};

$("restore").onclick = async () => {
  const session_id = localStorage.getItem("session_id");
  if (!session_id) return showError("Nothing to restore yet: sanitise some text first.");
  showError("");
  try {
    const r = await post("/restore", { text: $("reply").value, session_id });
    $("restored").value = r.restored_text;
    $("restored").hidden = $("copyRestored").hidden = false;
  } catch (e) {
    showError("Cannot reach the PrivyLens engine (is it running?).");
  }
};

$("copyRestored").onclick = () => navigator.clipboard.writeText($("restored").value);

pageSelection().then((t) => { if (t) $("input").value = t; });
