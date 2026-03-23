<<<<<<< HEAD
# Internship-projeccts
# Whisper Speech-to-Text GUI Project

## 📌 Description
This project is a Speech-to-Text desktop application built using Python, PyQt5 GUI, and OpenAI Whisper model.  
The application records speaker voice from microphone and converts it into text.

## 🚀 Features
- Real-time speech recognition
- GUI using PyQt5
- Whisper AI model for speech-to-text
- Simple user interface
- Internship practice project

## 🛠️ Technologies Used
- Python
- PyQt5
- OpenAI Whisper
- Torch
- SpeechRecognition

## 📂 Project Structure
whisper-gui-project/
│── main.py
│── gui.py
│── requirements.txt
│── README.md

## ▶️ How to Run

1. Install Python
2. Install libraries

pip install -r requirements.txt

3. Run the project

python main.py

## 📌 Author
Gowtham M

## 📌 Project Type
Internship Project / Practice Project
=======
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
>>>>>>> 5a1eacf4fa725094fe612cf5d556de9846ce04b1
