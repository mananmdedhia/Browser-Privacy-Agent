// AUTO-PROTECT: runs inside AI chat sites (see "matches" in manifest.json).
//   PASTE   anything you paste into a chat box is sanitised first, so the site only ever receives placeholders.
//   REPLIES placeholders in the page ([PERSON_1]) are shown with their real values again. Display only.
// The placeholder -> real value table lives in this tab's memory and is never sent anywhere.
const PLACEHOLDER = /\[[A-Z_]+_\d+\]/g;
const EDITABLE = 'textarea, input[type="text"], [contenteditable="true"], [contenteditable=""], [contenteditable="plaintext-only"]';
const mapping = {};
let enabled = true, sessionId = null;

chrome.storage.local.get("enabled", (v) => { enabled = v.enabled !== false; });       // popup checkbox
chrome.storage.onChanged.addListener((c) => { if (c.enabled) enabled = c.enabled.newValue !== false; });

function toast(message, actionLabel, onAction) {
  document.getElementById("privylens-toast")?.remove();
  const box = document.createElement("div");
  box.id = "privylens-toast";
  box.style.cssText = "position:fixed;right:16px;bottom:16px;z-index:2147483647;max-width:340px;padding:10px 12px;" +
    "border-radius:8px;background:#1d2733;color:#fff;font:13px/1.4 system-ui,sans-serif;box-shadow:0 4px 16px rgba(0,0,0,.35)";
  box.textContent = "PrivyLens: " + message + " ";
  if (actionLabel) {
    const b = document.createElement("button");
    b.textContent = actionLabel;
    b.style.cssText = "margin-left:6px;padding:2px 8px;border-radius:5px;border:1px solid #8b98a8;background:transparent;color:#fff;cursor:pointer";
    b.onclick = () => { onAction(); box.remove(); };
    box.append(b);
  }
  document.body.append(box);
  setTimeout(() => box.remove(), 9000);
}

function insert(el, text) {
  el.focus();
  document.execCommand("insertText", false, text);          // works for plain boxes and rich editors, and keeps undo history
}

document.addEventListener("paste", async (e) => {
  const box = e.target.closest?.(EDITABLE);
  const text = e.clipboardData?.getData("text/plain");
  if (!enabled || !box || !text || text.trim().length < 3) return;
  e.preventDefault();                                       // we paste the cleaned version instead
  e.stopImmediatePropagation();
  const r = await chrome.runtime.sendMessage({ type: "sanitise", text, session_id: sessionId });
  if (!r) return toast("the local engine is not running, so the paste was blocked.", "Paste anyway", () => insert(box, text));
  if (r.blocked) return toast("could not make this text safe, so the paste was blocked.", "Paste anyway", () => insert(box, text));
  sessionId = r.session_id;
  const chars = Array.from(text);                           // engine offsets count code points, like Python
  r.spans.forEach((s) => { mapping[s.placeholder] = chars.slice(s.start, s.end).join(""); });
  insert(box, r.sanitised_text);
  if (r.spans.length) {
    const kinds = [...new Set(r.spans.map((s) => s.label))].join(", ");
    toast(`hid ${r.spans.length} private value(s) (${kinds}).`, "Undo", () => { document.execCommand("undo"); insert(box, text); });
  }
}, true);                                                   // capture phase: run before the site's own paste handler

function restore(node) {
  if (node.nodeType === Node.TEXT_NODE) {
    if (node.parentElement?.closest(EDITABLE)) return;      // never un-hide inside the box the user is typing in
    if (node.nodeValue.includes("[")) node.nodeValue = node.nodeValue.replace(PLACEHOLDER, (m) => mapping[m] ?? m);
  } else node.childNodes.forEach(restore);
}
new MutationObserver((records) => records.forEach((m) => {
  m.addedNodes.forEach(restore);
  if (m.type === "characterData") restore(m.target);
})).observe(document.body, { childList: true, subtree: true, characterData: true });
