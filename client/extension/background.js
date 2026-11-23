// VaaniShield service worker: routes requests, manages the offscreen recorder, shows results on the page.
const DEFAULTS = { server: "http://127.0.0.1:8000", model: "auto" };
const MAX_SECONDS = 300; // safety cap; the user normally presses Stop

async function settings() {
  return { ...DEFAULTS, ...(await chrome.storage.sync.get(Object.keys(DEFAULTS))) };
}

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "vaani-check-media",
    title: "Check this voice with VaaniShield",
    contexts: ["audio", "video", "link"],
  });
  chrome.contextMenus.create({
    id: "vaani-scan-tab",
    title: "VaaniShield: scan this tab's audio",
    contexts: ["page", "frame", "selection", "image"],
  });
});

// ---------- state shared with the popup ----------
async function setState(state) {
  await chrome.storage.local.set({ vaaniState: { ...state, at: Date.now() } });
  chrome.runtime.sendMessage({ type: "state", state }).catch(() => {});
}

function setBadge(tabId, label) {
  const map = { FAKE: ["FAKE", "#f85149"], REAL: ["REAL", "#2ea043"], REC: ["REC", "#d29922"], ERR: ["ERR", "#6e7681"] };
  const [text, color] = map[label] || ["", "#000"];
  chrome.action.setBadgeText({ tabId, text });
  chrome.action.setBadgeBackgroundColor({ tabId, color });
}

// ---------- on-page overlay ----------
async function showOnPage(tabId, payload) {
  try {
    await chrome.scripting.executeScript({ target: { tabId }, func: vaaniToast, args: [payload] });
  } catch (e) { /* chrome:// pages etc. */ }
}

// Injected into the page (must be self-contained).
function vaaniToast(p) {
  let box = document.getElementById("__vaanishield_toast");
  if (!box) {
    box = document.createElement("div");
    box.id = "__vaanishield_toast";
    box.style.cssText = "position:fixed;z-index:2147483647;right:20px;bottom:20px;width:320px;font:14px/1.45 system-ui,Segoe UI,sans-serif;" +
      "background:#161b22;color:#e6edf3;border:1px solid #30363d;border-radius:14px;box-shadow:0 10px 30px #0008;padding:14px 16px;";
    document.documentElement.appendChild(box);
  }
  const close = '<span id="__vs_x" style="float:right;cursor:pointer;color:#8b949e;font-size:16px">✕</span>';
  const head = '<div style="font-weight:700;margin-bottom:8px">🛡️ Vaani<span style="color:#4f8cff">Shield</span>' + close + "</div>";
  clearInterval(window.__vsTimer);
  if (p.kind === "recording") {
    box.innerHTML = head + `<div>⏺ Listening to this tab… <b id="__vs_t">0:00</b></div>
      <div style="color:#8b949e;font-size:12.5px;margin-top:4px">Play the voice now. Stop when it has finished.</div>
      <button id="__vs_stop" style="margin-top:12px;width:100%;padding:10px;border:0;border-radius:10px;background:#e5484d;color:#fff;font:600 14px system-ui,Segoe UI,sans-serif;cursor:pointer">■ Stop &amp; check</button>`;
    const t0 = Date.now();
    window.__vsTimer = setInterval(() => {
      const s = Math.floor((Date.now() - t0) / 1000), el = document.getElementById("__vs_t");
      if (el) el.textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
    }, 1000);
    box.querySelector("#__vs_stop").onclick = () => { chrome.runtime.sendMessage({ type: "stop-recording" }); };
  } else if (p.kind === "analyzing") {
    box.innerHTML = head + "<div>⏳ Analyzing voice…</div>";
  } else if (p.kind === "error") {
    box.innerHTML = head + `<div style="color:#f85149">⚠ ${p.message}</div>`;
  } else {
    const r = p.result, fake = r.label === "FAKE", c = fake ? "#f85149" : "#2ea043";
    box.innerHTML = head +
      `<div style="display:flex;align-items:center;gap:12px">
         <div style="font-size:22px;font-weight:800;color:${c};border:1px solid ${c};border-radius:10px;padding:4px 12px;background:${c}22">${r.label}</div>
         <div><div style="font-weight:600">${r.verdict}</div><div style="color:#8b949e;font-size:12.5px">${r.confidence}% confidence</div></div>
       </div>
       <div style="height:8px;border-radius:99px;margin-top:12px;background:#30363d;overflow:hidden"><div style="height:100%;width:${r.fake_probability}%;background:${c}"></div></div>
       <div style="display:flex;justify-content:space-between;color:#8b949e;font-size:12px;margin-top:4px"><span>Real</span><span>${r.fake_probability}% fake</span><span>Fake</span></div>
       <div style="color:#8b949e;font-size:11.5px;margin-top:8px">${r.duration_s}s analysed · ${r.latency_ms} ms</div>`;
  }
  box.querySelector("#__vs_x").onclick = () => box.remove();
}

// ---------- server call ----------
async function analyzeBlob(blob, filename) {
  const s = await settings();
  const fd = new FormData();
  fd.append("file", blob, filename);
  fd.append("model", s.model);
  const r = await fetch(`${s.server}/api/analyze`, { method: "POST", body: fd });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.detail || `server error ${r.status}`);
  return j;
}

async function finish(tabId, result, source) {
  setBadge(tabId, result.label);
  await setState({ phase: "done", result, source });
  await showOnPage(tabId, { kind: "result", result });
}

async function fail(tabId, message) {
  setBadge(tabId, "ERR");
  await setState({ phase: "error", message });
  await showOnPage(tabId, { kind: "error", message });
}

// ---------- tab audio capture via offscreen document ----------
async function ensureOffscreen() {
  const ctx = await chrome.runtime.getContexts({ contextTypes: ["OFFSCREEN_DOCUMENT"] });
  if (ctx.length) return;
  await chrome.offscreen.createDocument({
    url: "offscreen.html",
    reasons: ["USER_MEDIA"],
    justification: "Record a few seconds of tab audio to check for AI-generated voices.",
  });
}

let busyTab = null;

async function scanTab(tabId) {
  if (busyTab !== null) return;
  try {
    busyTab = tabId;
    const streamId = await chrome.tabCapture.getMediaStreamId({ targetTabId: tabId });
    await ensureOffscreen();
    setBadge(tabId, "REC");
    await setState({ phase: "recording", tabId, startedAt: Date.now() });
    await showOnPage(tabId, { kind: "recording" });
    chrome.runtime.sendMessage({ type: "offscreen-record", streamId, tabId, maxSeconds: MAX_SECONDS });
  } catch (e) {
    busyTab = null;
    await fail(tabId, "Could not capture tab audio: " + e.message);
  }
}

async function checkUrl(tabId, url) {
  try {
    await setState({ phase: "analyzing" });
    await showOnPage(tabId, { kind: "analyzing" });
    const r = await fetch(url);
    if (!r.ok) throw new Error(`download failed (${r.status})`);
    const blob = await r.blob();
    const name = (new URL(url).pathname.split("/").pop() || "media") + "";
    await finish(tabId, await analyzeBlob(blob, name.includes(".") ? name : name + ".mp4"), "link");
  } catch (e) {
    // streamed media (Instagram, YouTube…) cannot be downloaded; fall back to recording the tab
    await scanTab(tabId);
  }
}

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (!tab?.id) return;
  if (info.menuItemId === "vaani-check-media") {
    const url = info.srcUrl || info.linkUrl;
    if (url && /^https?:/.test(url)) checkUrl(tab.id, url);
    else scanTab(tab.id); // blob:/MediaSource players
  } else if (info.menuItemId === "vaani-scan-tab") {
    scanTab(tab.id);
  }
});

chrome.runtime.onMessage.addListener((msg, sender, reply) => {
  if (msg.type === "scan-tab") {
    scanTab(msg.tabId);
  } else if (msg.type === "stop-recording") {
    chrome.runtime.sendMessage({ type: "offscreen-stop" }).catch(() => {});
  } else if (msg.type === "recorded") {
    // from offscreen: audio is ready (base64), send to server
    (async () => {
      const tabId = msg.tabId;
      try {
        await setState({ phase: "analyzing" });
        await showOnPage(tabId, { kind: "analyzing" });
        const bytes = Uint8Array.from(atob(msg.data), (c) => c.charCodeAt(0));
        const result = await analyzeBlob(new Blob([bytes], { type: msg.mime }), "tab-audio.webm");
        await finish(tabId, result, "tab");
      } catch (e) {
        await fail(tabId, e.message.includes("Failed to fetch") ? "VaaniShield server is offline. Start it on your laptop." : e.message);
      } finally {
        busyTab = null;
      }
    })();
  } else if (msg.type === "record-error") {
    busyTab = null;
    fail(msg.tabId, msg.message);
  } else if (msg.type === "analyze-file") {
    (async () => {
      try {
        const bytes = Uint8Array.from(atob(msg.data), (c) => c.charCodeAt(0));
        const result = await analyzeBlob(new Blob([bytes], { type: msg.mime }), msg.name);
        if (msg.tabId) await finish(msg.tabId, result, "file");
        else await setState({ phase: "done", result, source: "file" });
      } catch (e) {
        await setState({ phase: "error", message: e.message.includes("Failed to fetch") ? "VaaniShield server is offline." : e.message });
      }
    })();
  }
  return false;
});
