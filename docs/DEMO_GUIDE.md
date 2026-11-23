# VaaniShield — demo guide

## What you are showing
- **Backend** (`server/app.py`): FastAPI server on the laptop GPU. Auto mode uses the **multilingual detector**:
  NII AntiDeepfake MMS-300M (post-trained on ~74k h, 100+ languages) + VaaniShield fine-tune
  (top 4 layers, Hindi FLEURS vs MMS-TTS, Hinglish pilot, English replay).
  - Error rate (EER), clips never trained on: English 0.4 %, Hindi 0.0 %*, Hinglish 0.0 %.
    *Hindi fakes come from the same TTS family as training, so this is optimistic. Zero-shot (before
    our fine-tune): English 2.0 %, Hindi 1.0 %, Hinglish 0.0 %. These stricter numbers are on the website.
  - Comparison models: AASIST (English-only): 0.82 % English, 52 % Hindi, 58 % Hinglish;
    LFCC-CNN (ours): 20.8 % / 17.9 % / 13.3 %. Chart: `runs/demo/language_gap.png`.
  - Robustness: correct on Opus 128k / Opus 24k / MP3 for 13/14 tested clips; the miss is attack A19
    under MP3 (`demo_clips/_hard_cases/`).
- **Web app** (`http://127.0.0.1:8000`): upload a file or record from the microphone, then see REAL/FAKE, the confidence, a timeline and the model accuracy table.
- **Browser extension** (`client/extension/`): works on **any** website (WhatsApp Web, Instagram, YouTube, X, Facebook…).
  - Popup → **Start listening to this tab**, play the voice, then press **Stop & check** (in the popup or on the page). The verdict appears on the page.
  - Right-click a video/audio → **Check this voice with VaaniShield**.
  - Popup → **Check an audio/video file**.

## 1. Start the backend
Double-click `start_demo.bat`. This opens the server window and, about 12 s later, the browser.
Keep the black server window open.

Or start it by hand:
```
.venv\Scripts\python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
```
Check: http://127.0.0.1:8000/api/health → `{"status":"ok","device":"cuda",...}`

## 2. Load the extension (one time)
1. Open Chrome (or Edge) and go to `chrome://extensions`.
2. Turn on **Developer mode** (top right).
3. Click **Load unpacked** and choose the folder `C:\Users\happy\Desktop\Vanishield\client\extension`.
4. Pin it: puzzle-piece icon → pin **VaaniShield**.
5. Open the popup. It should say **● online**.

## 3. Demo script (about 5 minutes)
1. **Problem (30 s):** voice-cloning scams on WhatsApp and Instagram, especially in Hindi/Hinglish.
2. **Web app (1 min):** drop a real clip from `demo_clips/real/` → REAL. Drop a fake clip from `demo_clips/fake/` → FAKE, with the timeline. Show the accuracy table at the bottom.
3. **Extension on social media (2 min):**
   - Open a page that plays one of the demo clips, e.g. a YouTube video or WhatsApp Web voice note you prepared.
   - Click the extension → **Start listening** → play the clip → **Stop & check** → the verdict appears on the page.
   - Right-click a video → **Check this voice with VaaniShield**.
4. **Results (1 min):** `runs/demo/roc.png`, `runs/demo/eer_by_attack.png`, `runs/demo/results.md`.
5. **Honest limits and next steps (30 s):**
   - Trained on English ASVspoof 2019.
   - Next: Hindi/Hinglish data (the pipeline and data are ready), codec robustness for WhatsApp compression, and an XLS-R multilingual model.

## Rules for a safe demo
- **Rehearse every clip** you will play, through the same path (web app or extension), the day before.
- Test live microphone and new TTS voices (ElevenLabs etc.) beforehand. They are *out of domain* and can be wrong. If you show them, say so.
- Record a backup screen video of the full demo (Win + Alt + R with Xbox Game Bar).
- Everything runs offline on the laptop, so no internet is needed once the server is running.
- Say "balanced accuracy" or "EER", not plain "accuracy". 90 % of the test clips are fake, so plain accuracy flatters any model.

## Troubleshooting
| Problem | Fix |
|---|---|
| Popup says **offline** | the server window is closed → run `start_demo.bat` again |
| "Could not capture tab audio" | click the extension icon *on that tab* first (Chrome needs the click), then scan; does not work on `chrome://` pages |
| Result "no speech detected" | nothing was playing while listening → start, play the clip, then stop |
| Tab goes silent while scanning | it should not (audio is routed back); reload the extension at `chrome://extensions` |
