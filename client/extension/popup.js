const $ = (id) => document.getElementById(id);
const DEFAULTS = { server: "http://127.0.0.1:8000" };

async function init() {
  const s = { ...DEFAULTS, ...(await chrome.storage.sync.get(Object.keys(DEFAULTS))) };
  $("server").value = s.server;
  $("server").addEventListener("change", () => chrome.storage.sync.set({ server: $("server").value.trim().replace(/\/$/, "") }).then(ping));
  ping();
  const { vaaniState } = await chrome.storage.local.get("vaaniState");
  if (vaaniState && Date.now() - vaaniState.at < 10 * 60 * 1000) render(vaaniState);
}

async function ping() {
  const server = $("server").value.trim().replace(/\/$/, "");
  try {
    const h = await (await fetch(`${server}/api/health`, { signal: AbortSignal.timeout(3000) })).json();
    $("status").textContent = `● online · ${h.device.toUpperCase()}`; $("status").className = "status ok";
  } catch { $("status").textContent = "● offline"; $("status").className = "status bad"; }
}

async function activeTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab;
}

let recording = false;
$("scan").addEventListener("click", async () => {
  if (recording) { chrome.runtime.sendMessage({ type: "stop-recording" }); return; }
  const tab = await activeTab();
  if (!tab || /^(chrome|edge|about):/.test(tab.url || "")) { render({ phase: "error", message: "Open a website (e.g. WhatsApp Web, Instagram, YouTube) first." }); return; }
  chrome.runtime.sendMessage({ type: "scan-tab", tabId: tab.id });
});

$("file").addEventListener("change", async (e) => {
  const f = e.target.files[0]; if (!f) return;
  render({ phase: "analyzing" });
  const buf = new Uint8Array(await f.arrayBuffer());
  let bin = "";
  for (let i = 0; i < buf.length; i += 0x8000) bin += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
  const tab = await activeTab();
  chrome.runtime.sendMessage({ type: "analyze-file", data: btoa(bin), mime: f.type, name: f.name, tabId: tab?.id });
});

chrome.runtime.onMessage.addListener((msg) => { if (msg.type === "state") render(msg.state); });

function render(st) {
  const out = $("out"); out.hidden = false;
  recording = st.phase === "recording";
  $("scan").disabled = st.phase === "analyzing";
  $("scan").textContent = recording ? "■ Stop & check" : "🎧 Start listening to this tab";
  $("scan").style.background = recording ? "var(--fake)" : "";
  if (st.phase === "recording") out.innerHTML = `<span class="spinner"></span>Listening… play the voice, then press Stop &amp; check.`;
  else if (st.phase === "analyzing") out.innerHTML = `<span class="spinner"></span>Analyzing voice…`;
  else if (st.phase === "error") out.innerHTML = `<div class="err">⚠ ${st.message}</div>`;
  else if (st.phase === "done") {
    const r = st.result, c = r.label === "FAKE" ? "var(--fake)" : "var(--real)";
    out.innerHTML = `<div class="row"><div class="badge ${r.label}">${r.label}</div>
      <div><div><b>${r.verdict}</b></div><div class="small">${r.confidence}% confidence</div></div></div>
      <div class="bar"><div style="width:${r.fake_probability}%;background:${c}"></div></div>
      <div class="lbl"><span>Real</span><span>${r.fake_probability}% fake</span><span>Fake</span></div>
      <div class="small">${r.duration_s}s analysed · ${r.latency_ms} ms</div>`;
  }
}

init();
