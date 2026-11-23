const $ = (id) => document.getElementById(id);
let clip = null, clipName = null;
let rec = null, recStream = null, recTimer = null;

async function init() {
  try {
    await (await fetch("/api/health")).json();
    $("status").textContent = "detector online"; $("status").className = "status-dot ok";
  } catch {
    $("status").textContent = "detector offline — run start_demo.bat"; $("status").className = "status-dot bad";
  }
}

function setClip(b, name) {
  clip = b; clipName = name;
  $("player").src = URL.createObjectURL(b); $("player").hidden = false;
  $("dropText").textContent = name; $("go").disabled = false;
}

$("file").onchange = (e) => { const f = e.target.files[0]; if (f) setClip(f, f.name); };
const drop = $("drop");
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); const f = e.dataTransfer.files[0]; if (f) setClip(f, f.name); };

const fmt = (s) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;

// Start / stop recording — no fixed length; the user decides when to stop.
$("rec").onclick = async () => {
  const btn = $("rec");
  if (rec && rec.state === "recording") { rec.stop(); return; }
  try {
    recStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
  } catch (e) { alert("Microphone not available: " + e.message); return; }
  const chunks = [];
  rec = new MediaRecorder(recStream);
  rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  rec.onstop = () => {
    clearInterval(recTimer);
    recStream.getTracks().forEach((t) => t.stop());
    btn.classList.remove("recording"); btn.textContent = "● Record again";
    $("recHint").textContent = "Recorded. Press “Check this voice”.";
    setClip(new Blob(chunks, { type: rec.mimeType }), "your recording.webm");
  };
  rec.start(250);
  let secs = 0;
  btn.classList.add("recording"); btn.textContent = "■ Stop  0:00";
  $("recHint").textContent = "Recording… speak now. Press stop when you are done.";
  recTimer = setInterval(() => {
    secs++; btn.textContent = `■ Stop  ${fmt(secs)}`;
    if (secs >= 600) rec.stop(); // safety cap: 10 minutes
  }, 1000);
};

// ---------- pipeline animation (real stages; details filled from the server's measurements) ----------
const STEPS = ["upload", "decode", "slice", "model", "verdict"];
const pipe = $("pipeline");
const li = (k) => pipe.querySelector(`[data-step="${k}"]`);
const detail = (k, t) => (li(k).querySelector("[data-detail]").textContent = t);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
function setStep(k, state) {
  const el = li(k);
  el.classList.remove("active", "done", "fake", "real");
  if (state) el.classList.add(...state.split(" "));
  const i = STEPS.indexOf(k);
  const done = STEPS.filter((s) => li(s).classList.contains("done")).length;
  pipe.style.setProperty("--prog", `${Math.min(80, (Math.max(done, state?.includes("active") ? i : 0) / 4) * 80)}%`);
}
function resetPipeline() {
  pipe.hidden = false;
  STEPS.forEach((k) => setStep(k, ""));
  detail("upload", "sending audio"); detail("decode", "to 16 kHz mono"); detail("slice", "into short windows");
  detail("model", "listens for AI traces"); detail("verdict", "AI or human");
  li("upload").querySelector(".bar i").style.width = "0%";
}
const kb = (n) => (n > 1048576 ? (n / 1048576).toFixed(1) + " MB" : Math.max(1, Math.round(n / 1024)) + " KB");

function post(fd, onUpload) {
  return new Promise((resolve, reject) => {
    const x = new XMLHttpRequest();
    x.open("POST", "/api/analyze");
    x.upload.onprogress = (e) => e.lengthComputable && onUpload(e.loaded / e.total);
    x.upload.onload = () => onUpload(1, true);
    x.onload = () => {
      let j = {};
      try { j = JSON.parse(x.responseText); } catch {}
      x.status < 300 ? resolve(j) : reject(new Error(j.detail || x.statusText));
    };
    x.onerror = () => reject(new Error("server not reachable"));
    x.send(fd);
  });
}

$("go").onclick = async () => {
  if (!clip) return;
  const go = $("go"); go.disabled = true; go.innerHTML = `<span class="spinner"></span> Analysing…`;
  $("result").classList.remove("show");
  resetPipeline();
  setStep("upload", "active");
  let uploaded = false, waitingSteps;
  const fd = new FormData(); fd.append("file", clip, clipName); fd.append("model", "auto");
  const req = post(fd, (f, finished) => {
    li("upload").querySelector(".bar i").style.width = Math.round(f * 100) + "%";
    detail("upload", `${Math.round(f * 100)}% of ${kb(clip.size)}`);
    if (finished && !uploaded) {
      uploaded = true;
      setStep("upload", "done");
      // the server now decodes, slices and runs the network, in that order
      waitingSteps = (async () => {
        setStep("decode", "active"); await sleep(350);
        setStep("decode", "done"); setStep("slice", "active"); await sleep(300);
        setStep("slice", "done"); setStep("model", "active");
      })();
    }
  });
  try {
    const j = await req;
    if (!uploaded) { uploaded = true; setStep("upload", "done"); }
    await waitingSteps;
    const st = j.stages || {};
    detail("upload", `${kb(st.upload?.bytes || clip.size)} ${st.upload?.format || ""}`.trim());
    setStep("upload", "done");
    detail("decode", `${st.decode?.seconds ?? j.duration_s} s of audio · 16 kHz · ${st.decode?.ms ?? "–"} ms`);
    setStep("decode", "done");
    const sk = st.slice?.skipped_silent ? ` · ${st.slice.skipped_silent} silent skipped` : "";
    detail("slice", `${st.slice?.windows ?? j.n_windows} × ${st.slice?.window_s ?? 4} s slices${sk}`);
    setStep("slice", "done");
    await sleep(150);
    detail("model", `${st.model?.params_millions ?? "–"}M parameters · ${st.model?.ms ?? "–"} ms on ${st.model?.device ?? ""}`);
    setStep("model", "done");
    await sleep(200);
    const fake = j.label === "FAKE";
    detail("verdict", fake ? `AI-generated · ${j.confidence}%` : `Human · ${j.confidence}%`);
    setStep("verdict", "done " + (fake ? "fake" : "real"));
    pipe.style.setProperty("--prog", "80%");
    show(j);
  } catch (e) {
    STEPS.forEach((k) => li(k).classList.remove("active"));
    alert("Could not check this clip: " + e.message);
  }
  go.disabled = false; go.innerHTML = `Check this voice <span class="arrow">→</span>`;
};

function show(j) {
  const fake = j.label === "FAKE", ai = j.fake_probability, hu = Math.round((100 - ai) * 10) / 10;
  $("result").classList.remove("show"); void $("result").offsetWidth; $("result").classList.add("show");
  $("label").textContent = fake ? "Likely AI-generated" : "Likely a real voice";
  $("label").className = "label " + j.label;
  $("why").textContent = `${j.confidence}% confident`;
  requestAnimationFrame(() => ($("arc").style.strokeDashoffset = 389.6 * (1 - ai / 100)));
  $("gpct").textContent = Math.round(ai) + "%";
  $("sbAi").style.flexBasis = `${Math.max(ai, 0.0001)}%`; $("sbHu").style.flexBasis = `${Math.max(hu, 0.0001)}%`;
  $("sbAiPct").textContent = Math.round(ai) + "%"; $("sbHuPct").textContent = Math.round(hu) + "%";
  $("sbAi").style.paddingLeft = ai < 18 ? "0" : ""; $("sbHu").style.paddingRight = hu < 18 ? "0" : "";
  $("sbAi").querySelector("span").style.visibility = ai < 18 ? "hidden" : "";
  $("sbHu").querySelector("span").style.visibility = hu < 18 ? "hidden" : "";
  $("timeline").innerHTML = j.timeline.map((w) => {
    const p = w.fake_probability;
    return `<div class="bar" style="height:${Math.max(6, p)}%;background:${p >= 50 ? "#e5484d" : "#0e9f6e"}" data-tip="${w.start_s}–${w.end_s} s · ${Math.round(p)}% AI"></div>`;
  }).join("");
  $("tlStart").textContent = "0 s"; $("tlEnd").textContent = `${j.duration_s} s`;
  $("facts").innerHTML = [["Audio", j.duration_s + " s"], ["Slices checked", j.n_windows], ["Total time", j.latency_ms + " ms"],
    ["Model", "Multilingual · 100+ languages"]].map(([k, v]) => `<div>${k}<b>${v}</b></div>`).join("");
  $("note").textContent = j.note;
  setTimeout(() => $("result").scrollIntoView({ behavior: "smooth", block: "start" }), 250);
}

init();
