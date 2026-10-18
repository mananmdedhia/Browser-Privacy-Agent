// Two jobs, both local only (nothing here talks to the internet).

// 1) Content scripts run with the web page's identity, and the page is not allowed to call the local engine.
//    This worker has permission (host_permissions in manifest.json), so it makes the call for them.
chrome.runtime.onMessage.addListener((msg, _sender, reply) => {
  if (msg.type !== "sanitise") return;
  fetch("http://127.0.0.1:8000/sanitise", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: msg.text, session_id: msg.session_id }),
  })
    .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
    .then(reply, () => reply(null));          // null = engine not running
  return true;                                // keep the message channel open for the async reply
});

// 2) The shortcut (Alt+Shift+P) works on ANY page. Pressing it grants temporary access to just that tab (activeTab),
//    so we inject content.js there on demand instead of asking for permission to read every website.
chrome.commands.onCommand.addListener(async (command, tab) => {
  if (command !== "sanitise-field" || !tab) return;
  try {
    await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ["shared.js", "content.js"] });
    chrome.tabs.sendMessage(tab.id, { type: "sanitise-field" });
  } catch (e) { /* pages Chrome never allows extensions on (chrome://, the web store) */ }
});
