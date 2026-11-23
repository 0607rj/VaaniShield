// Offscreen document: records N seconds of a tab's audio (the user keeps hearing it), returns it as base64 WebM/Opus.
let current = null; // active MediaRecorder

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "offscreen-record") record(msg);
  else if (msg.type === "offscreen-stop" && current && current.state !== "inactive") current.stop();
});

async function record({ streamId, tabId, maxSeconds }) {
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { mandatory: { chromeMediaSource: "tab", chromeMediaSourceId: streamId } },
      video: false,
    });
  } catch (e) {
    chrome.runtime.sendMessage({ type: "record-error", tabId, message: "Tab audio capture failed: " + e.message });
    return;
  }
  // capturing mutes the tab by default; route it back to the speakers
  const ctx = new AudioContext();
  ctx.createMediaStreamSource(stream).connect(ctx.destination);

  const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : "audio/webm";
  const rec = new MediaRecorder(stream, { mimeType: mime, audioBitsPerSecond: 128000 });
  const chunks = [];
  rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  rec.onstop = async () => {
    stream.getTracks().forEach((t) => t.stop());
    ctx.close();
    const blob = new Blob(chunks, { type: mime });
    const buf = new Uint8Array(await blob.arrayBuffer());
    let bin = "";
    for (let i = 0; i < buf.length; i += 0x8000) bin += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
    chrome.runtime.sendMessage({ type: "recorded", tabId, mime, data: btoa(bin) });
  };
  current = rec;
  rec.start(250);
  setTimeout(() => rec.state !== "inactive" && rec.stop(), maxSeconds * 1000); // safety cap only
}
