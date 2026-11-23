# Your own Hinglish clips (makes the demo work on YOUR voice)

Put 4–15 second clips here, any format (m4a, mp3, ogg, wav, webm):

- `real/`  — 10–20 clips of real people speaking Hinglish: you and friends, recorded on the
  same phone/laptop mic you will use in the demo. Different sentences, some quiet, some noisy.
- `fake/`  — 10–20 AI voices speaking Hinglish: any TTS / voice-clone tool (e.g. a free online
  Hindi TTS, or a voice-clone app), saved as audio.

Rules
- Use DIFFERENT sentences from the clips you will play in the live demo (no training on the demo itself).
- Only record people who agreed to it.

Then tell Claude "retrain with my recordings" — it runs:
    python scripts/finetune_aasist_hinglish.py --device cuda --resplit --extra own_recordings --tag AASIST_hinglish_own
(about 10–15 minutes), and the server picks the new model up after a restart.
