// Helpers used by BOTH the popup and the in-page script (content.js), so highlights look and behave the same.
// All styling is inline, so it works inside any web page without a stylesheet.
const PL = {
  // The engine counts characters like Python (code points); JS strings count UTF-16 units.
  // Array.from(text) splits by code points, so offsets line up even with emoji.
  slice: (chars, a, b) => chars.slice(a, b).join(""),

  hue(label) {                                     // same label -> same colour
    let h = 0;
    for (const c of label) h = (h * 31 + c.charCodeAt(0)) % 360;
    return h;
  },

  // engine response -> list of spans we can switch on/off
  prepare(r) {
    const unsure = new Set(r.flagged.map((s) => s.start + ":" + s.end));
    return r.spans.map((s) => ({ ...s, on: true, flagged: unsure.has(s.start + ":" + s.end) }));
  },

  // the text with every "on" span replaced by its placeholder
  outputText(chars, spans) {
    let out = "", pos = 0;
    for (const s of spans) {
      out += PL.slice(chars, pos, s.start) + (s.on ? s.placeholder : PL.slice(chars, s.start, s.end));
      pos = s.end;
    }
    return out + PL.slice(chars, pos);
  },

  // draw the original text with highlighted spans; clicking one toggles it and calls onChange.
  // textContent only: page text is untrusted, so never innerHTML.
  renderMarked(box, chars, spans, onChange) {
    box.textContent = "";
    let pos = 0;
    for (const s of spans) {
      box.append(PL.slice(chars, pos, s.start));
      const el = document.createElement("span");
      el.style.cssText = "border-radius:3px;padding:0 2px;cursor:pointer;" +
        (s.on ? `background:hsl(${PL.hue(s.label)} 80% 85%);color:#1d2733;` : "text-decoration:line-through;opacity:.6;outline:1px dotted currentColor;") +
        (s.flagged ? "outline:2px dashed #e3b341;outline-offset:1px;" : "");
      el.title = `${s.label} -> ${s.placeholder} (confidence ${s.conf.toFixed(2)}, found by ${s.source}). Click to ${s.on ? "keep visible" : "hide again"}`;
      el.append(PL.slice(chars, s.start, s.end));
      const tag = document.createElement("small");
      tag.textContent = s.label;
      tag.style.cssText = "font-size:10px;opacity:.7;margin-left:3px";
      el.append(tag);
      el.onclick = () => { s.on = !s.on; onChange(); };
      box.append(el);
      pos = s.end;
    }
    box.append(PL.slice(chars, pos));
  },
};
