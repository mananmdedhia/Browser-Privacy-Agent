// The in-page half of PrivyLens. It runs on the AI chat sites (manifest.json), on any site you switch on with the
// popup's "Protect this site" button, and for one moment on any page when you press the shortcut (Alt+Shift+P).
//   PASTE     anything you paste into a text box is cleaned first, so the site only ever receives placeholders.
//   SHORTCUT  cleans what you selected in the text box, or the whole box if nothing is selected (also fine for typed text).
//   REPLIES   placeholders on the page ([PERSON_1]) are shown with their real values again. Display only.
// Optional "ask before pasting" shows a preview you can edit. The placeholder -> real value table lives in this
// tab's memory and is never sent anywhere.
if (!window.__privylens) {              // the shortcut can inject this file again: only set up once per page
  window.__privylens = true;
  const PLACEHOLDER = /\[[A-Z_]+_\d+\]/g;
  const EDITABLE = 'textarea, input[type="text"], input[type="search"], input:not([type]), [contenteditable="true"], [contenteditable=""], [contenteditable="plaintext-only"]';
  const mapping = {};
  const cfg = { enabled: true, preview: false };       // switches in the popup
  let sessionId = null;

  chrome.storage.local.get(["enabled", "preview"], (v) => { cfg.enabled = v.enabled !== false; cfg.preview = v.preview === true; });
  chrome.storage.onChanged.addListener((c) => {
    if (c.enabled) cfg.enabled = c.enabled.newValue !== false;
    if (c.preview) cfg.preview = c.preview.newValue === true;
  });

  const make = (tag, style, text) => {
    const e = document.createElement(tag);
    e.style.cssText = style;
    if (text) e.textContent = text;
    return e;
  };
  const BTN = "padding:6px 12px;border-radius:6px;border:1px solid #8b98a8;background:transparent;color:#fff;cursor:pointer;font:inherit;";

  function toast(message, actionLabel, onAction) {
    document.getElementById("privylens-toast")?.remove();
    const box = make("div", "position:fixed;right:16px;bottom:16px;z-index:2147483647;max-width:340px;padding:10px 12px;" +
      "border-radius:8px;background:#1d2733;color:#fff;font:13px/1.4 system-ui,sans-serif;box-shadow:0 4px 16px rgba(0,0,0,.35)", "PrivyLens: " + message + " ");
    box.id = "privylens-toast";
    if (actionLabel) {
      const b = make("button", BTN + "margin-left:6px;padding:2px 8px", actionLabel);
      b.onclick = () => { onAction(); box.remove(); };
      box.append(b);
    }
    document.body.append(box);
    setTimeout(() => box.remove(), 9000);
  }

  const editableOf = (node) => node?.closest?.(EDITABLE) ?? null;
  const selectedText = (box) => typeof box.selectionStart === "number" ? box.value.slice(box.selectionStart, box.selectionEnd) : String(getSelection());

  function saveSelection(box) {          // the preview steals focus; this puts the cursor / selection back afterwards
    if (typeof box.selectionStart === "number") {
      const [a, b] = [box.selectionStart, box.selectionEnd];
      return () => { box.focus(); box.setSelectionRange(a, b); };
    }
    const sel = getSelection(), range = sel.rangeCount ? sel.getRangeAt(0).cloneRange() : null;
    return () => { box.focus(); if (range) { sel.removeAllRanges(); sel.addRange(range); } };
  }

  // execCommand works in plain boxes and rich editors alike, and keeps the browser's undo history
  const insert = (box, text) => { box.focus(); document.execCommand("insertText", false, text); };

  function preview(chars, spans, original, done) {
    const overlay = make("div", "position:fixed;inset:0;z-index:2147483647;background:rgba(0,0,0,.5);display:flex;align-items:center;justify-content:center");
    const card = make("div", "width:min(580px,92vw);max-height:88vh;overflow:auto;padding:16px;border-radius:10px;background:#1d2733;color:#fff;font:14px/1.5 system-ui,sans-serif");
    const marked = make("div", "white-space:pre-wrap;word-break:break-word;max-height:170px;overflow:auto;padding:8px;border:1px solid #3a4756;border-radius:6px");
    const output = make("div", "white-space:pre-wrap;word-break:break-word;max-height:170px;overflow:auto;padding:8px;border:1px solid #3a4756;border-radius:6px");
    const close = (value) => { overlay.remove(); document.removeEventListener("keydown", onKey, true); done(value); };
    const onKey = (e) => { if (e.key === "Escape") { e.stopPropagation(); close(null); } };
    const draw = () => { PL.renderMarked(marked, chars, spans, draw); output.textContent = PL.outputText(chars, spans); };
    const row = make("div", "display:flex;gap:8px;margin-top:12px;flex-wrap:wrap");
    const ok = make("button", BTN + "background:#1f6feb;border-color:#1f6feb", "Insert sanitised");
    const orig = make("button", BTN, "Insert original");
    const cancel = make("button", BTN, "Cancel");
    ok.onclick = () => close(PL.outputText(chars, spans));
    orig.onclick = () => close(original);
    cancel.onclick = () => close(null);
    row.append(ok, orig, cancel);
    card.append(make("div", "font-weight:600;font-size:15px", "PrivyLens: review before inserting"),
      make("div", "margin:10px 0 4px;color:#8b98a8", "Click a highlight to keep that value visible. Dashed = the model was unsure."), marked,
      make("div", "margin:10px 0 4px;color:#8b98a8", "What will be inserted:"), output, row);
    overlay.dataset.privylens = "1";          // the page-restore code must leave this panel alone: it shows placeholders on purpose
    overlay.append(card);
    document.body.append(overlay);
    document.addEventListener("keydown", onKey, true);
    draw();
    ok.focus();
  }

  // mode "paste": new text goes in at the cursor. mode "field": the text came from the box itself and is replaced.
  async function protect(box, text, mode) {
    const restore = saveSelection(box);
    const r = await chrome.runtime.sendMessage({ type: "sanitise", text, session_id: sessionId });
    const problem = (msg) => toast(msg, mode === "paste" ? "Paste anyway" : null, () => { restore(); insert(box, text); });
    if (!r) return problem("the local engine is not running" + (mode === "paste" ? ", so the paste was blocked." : "."));
    if (r.blocked) return problem("could not make this text safe" + (mode === "paste" ? ", so the paste was blocked." : "."));
    const chars = Array.from(text), spans = PL.prepare(r);
    const finish = (out) => {
      if (out === null) return;
      sessionId = r.session_id;
      spans.forEach((s) => { mapping[s.placeholder] = PL.slice(chars, s.start, s.end); });
      restore();
      insert(box, out);
      const hidden = spans.filter((s) => s.on);
      if (hidden.length) {
        toast(`hid ${hidden.length} private value(s) (${[...new Set(hidden.map((s) => s.label))].join(", ")}).`, "Undo",
          () => { document.execCommand("undo"); if (mode === "paste") insert(box, text); });
      }
    };
    if (cfg.preview) preview(chars, spans, text, finish); else finish(PL.outputText(chars, spans));
  }

  document.addEventListener("paste", (e) => {
    const box = editableOf(e.target), text = e.clipboardData?.getData("text/plain");
    if (!cfg.enabled || !box || !text || text.trim().length < 3) return;
    e.preventDefault();                  // we insert the cleaned version instead
    e.stopImmediatePropagation();
    protect(box, text, "paste");
  }, true);                              // capture phase: before the site's own paste handler

  chrome.runtime.onMessage.addListener((m) => {      // sent by background.js when the shortcut is pressed
    if (m.type !== "sanitise-field") return;
    const box = editableOf(document.activeElement);
    if (!box) return toast("click inside a text box first, then press the shortcut.");
    let text = selectedText(box);
    if (!text) {                         // nothing selected: take the whole box
      if (typeof box.select === "function") box.select(); else document.execCommand("selectAll");
      text = selectedText(box);
    }
    if (text.trim()) protect(box, text, "field");
  });

  function restoreValues(node) {
    if (node.nodeType === Node.TEXT_NODE) {
      if (node.parentElement?.closest(EDITABLE + ", [data-privylens]")) return;   // never un-hide in the box being typed in, or in our own preview
      if (!node.nodeValue.includes("[")) return;
      const next = node.nodeValue.replace(PLACEHOLDER, (m) => mapping[m] ?? m);
      if (next !== node.nodeValue) node.nodeValue = next;      // only write when changed: writing is itself a change the observer would see (endless loop)
    } else node.childNodes.forEach(restoreValues);
  }
  new MutationObserver((records) => records.forEach((m) => {
    m.addedNodes.forEach(restoreValues);
    if (m.type === "characterData") restoreValues(m.target);
  })).observe(document.body, { childList: true, subtree: true, characterData: true });
}
