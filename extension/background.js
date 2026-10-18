// Content scripts run with the web page's identity, and the page is not allowed to call the local engine.
// This background worker has permission (host_permissions in manifest.json), so it makes the call for them.
// Still local only: the request goes to 127.0.0.1, never to the internet.
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
