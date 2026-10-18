// PrivyLens popup: settings, "protect this site", and MANUAL mode (select text -> Sanitise).
// The automatic modes (paste cleaning, shortcut) live in content.js. Talks only to the engine on THIS computer.
//
// Use it:  1) start the engine:  python -m uvicorn api.server:app --host 127.0.0.1 --port 8000
//          2) chrome://extensions -> turn on Developer mode -> Load unpacked -> pick this `extension` folder
//          3) paste into ChatGPT / Claude, or click in any text box and press Alt+Shift+P.
const API = "http://127.0.0.1:8000";
const $ = (id) => document.getElementById(id);
const inExtension = typeof chrome !== "undefined" && !!chrome.storage;   // false when this page is opened as a plain web page
let state = null;                       // {chars, spans, blocked} for the text currently under review

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

// ---------- settings (content.js reads the same values) ----------
if (inExtension) {
  chrome.storage.local.get(["enabled", "preview"], (v) => { $("auto").checked = v.enabled !== false; $("previewOpt").checked = v.preview === true; });
}
$("auto").onchange = () => inExtension && chrome.storage.local.set({ enabled: $("auto").checked });
$("previewOpt").onchange = () => inExtension && chrome.storage.local.set({ preview: $("previewOpt").checked });

// ---------- "protect this site": switch on paste cleaning for the site in the current tab ----------
// Asks Chrome for permission for THIS site only, then registers content.js there. Built-in AI sites are always on.
async function setupSiteButton() {
  if (!inExtension || !chrome.permissions) return;
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  let u;
  try { u = new URL(tab.url); } catch { return; }
  if (!/^https?:$/.test(u.protocol)) return;
  const id = "site-" + u.host, pattern = u.origin + "/*", btn = $("site");
  btn.hidden = false;
  if (chrome.runtime.getManifest().content_scripts[0].matches.some((m) => m.startsWith(u.origin + "/"))) {
    btn.textContent = `${u.host}: protected (built in)`;
    btn.disabled = true;
    return;
  }
  const on = (await chrome.scripting.getRegisteredContentScripts({ ids: [id] })).length > 0;
  btn.textContent = on ? `Stop protecting ${u.host}` : `Protect pastes on ${u.host}`;
  btn.onclick = async () => {
    if (on) {
      await chrome.scripting.unregisterContentScripts({ ids: [id] });
      await chrome.permissions.remove({ origins: [pattern] });
      return setupSiteButton();
    }
    if (await chrome.permissions.request({ origins: [pattern] })) {           // must come straight from the click
      await chrome.scripting.registerContentScripts([{ id, matches: [pattern], js: ["shared.js", "content.js"], runAt: "document_idle" }]);
      btn.textContent = `Done. Reload ${u.host} to switch it on.`;
      btn.disabled = true;
    }
  };
}

// ---------- manual mode ----------
function render() {
  const { chars, spans, blocked } = state;
  PL.renderMarked($("marked"), chars, spans, render);
  $("output").value = PL.outputText(chars, spans);

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
    state = { chars: Array.from(text), blocked: r.blocked, spans: PL.prepare(r) };
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

setupSiteButton();
pageSelection().then((t) => { if (t) $("input").value = t; });
