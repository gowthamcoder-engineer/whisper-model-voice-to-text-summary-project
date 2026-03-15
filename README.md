# whisper-model-voice-to-text-summary-project
Internship projects on Speech-to-Text, Whisper AI, GUI development, and audio processing using Python.
# SpeakSense Desktop GUI 🎙

## Setup

```bash
# 1. Install FFmpeg
brew install ffmpeg          # macOS
sudo apt install ffmpeg      # Ubuntu

# 2. Ollama
ollama serve
ollama pull llama3

# 3. Python deps
pip install -r requirements.txt

# 4. Run
python speaksense.py
```

## GUI Layout

```
┌─────────────┬──────────────────┬──────────────────┬─────────────────┐
│  TIMELINE   │   SPEAKER 1      │   SPEAKER 2      │   SPEAKER 3     │
├─────────────┼──────────────────┼──────────────────┼─────────────────┤
│ 00:00→00:08 │ "Hello every..." │                  │                 │
│ 00:10→00:18 │                  │ "Yes I agree..." │                 │
│ 00:20→00:28 │                  │                  │ "Let me add..." │
│ ...         │ ...              │ ...              │ ...             │
├─────────────┼──────────────────┼──────────────────┼─────────────────┤
│ SUMMARY     │ Spk 1 summary... │ Spk 2 summary... │ Spk 3 summary.. │
├─────────────┴──────────────────┴──────────────────┴─────────────────┤
│  ✅ OVERALL BEST SUMMARY — Key points from all speakers             │
└─────────────────────────────────────────────────────────────────────┘
```

Excel is saved **automatically** to `exports/` when analysis completes.
