import os
import re
import sys
import wave
import uuid
import time
import queue
import struct
import threading
import datetime
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog as fd
import tkinter.messagebox as mb

import numpy as np
import customtkinter as ctk
from PIL import Image, ImageDraw

# ─────────────────────────────────────────────────────────────────────────────
# Theme
# ─────────────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

PALETTE = {
    "bg":       "#080D18",
    "surface":  "#0D1526",
    "card":     "#111D33",
    "raised":   "#152133",
    "border":   "#1C2F46",
    "text":     "#CCD9EC",
    "muted":    "#4A6680",
    "red":      "#FF4757",
    "green":    "#2ED573",
    "blue":     "#3D9BFF",
    "purple":   "#8B5CF6",
    "amber":    "#F59E0B",
    "teal":     "#14B8A6",
}

SPEAKER_COLORS = {
    "SPEAKER_01": "#4FC3F7",
    "SPEAKER_02": "#81C784",
    "SPEAKER_03": "#FFB74D",
    "SPEAKER_04": "#F48FB1",
    "SPEAKER_05": "#CE93D8",
    "SPEAKER_06": "#80CBC4",
}

SPEAKER_BG = {
    "SPEAKER_01": "#0D2B36",
    "SPEAKER_02": "#0D2B14",
    "SPEAKER_03": "#2B1D0A",
    "SPEAKER_04": "#2B0D1C",
    "SPEAKER_05": "#1D0D2B",
    "SPEAKER_06": "#0D2B28",
}

def spk_color(spk: str) -> str:
    return SPEAKER_COLORS.get(spk, "#90CAF9")

def spk_label(spk: str) -> str:
    n = int(spk.replace("SPEAKER_", ""), 10)
    return f"Speaker {n}"

def fmt_time(s: float) -> str:
    m = int(s) // 60
    sec = int(s) % 60
    return f"{m:02d}:{sec:02d}"

# ─────────────────────────────────────────────────────────────────────────────
# Audio Processor (Whisper + Diarization)
# ─────────────────────────────────────────────────────────────────────────────
class AudioProcessor:
    def __init__(self, model_size: str = "base"):
        import whisper
        self.model = whisper.load_model(model_size)
        self.model_size = model_size

    def transcribe_and_diarize(self, wav_path: str) -> Dict:
        result = self.model.transcribe(str(wav_path), word_timestamps=True, verbose=False)
        segments = []
        for seg in result["segments"]:
            segments.append({
                "id":      seg["id"],
                "start":   round(seg["start"], 2),
                "end":     round(seg["end"],   2),
                "text":    seg["text"].strip(),
                "speaker": "SPEAKER_01",
            })
        segments = self._heuristic_diarize(segments)
        speaker_data = self._build_speaker_data(segments)
        duration = segments[-1]["end"] if segments else 0
        full_text = " ".join(s["text"] for s in segments)
        return {
            "segments":     segments,
            "speaker_data": speaker_data,
            "full_text":    full_text,
            "duration":     duration,
            "language":     result.get("language", "en"),
        }

    def _heuristic_diarize(self, segments: List[Dict]) -> List[Dict]:
        SPEAKERS = ["SPEAKER_01", "SPEAKER_02", "SPEAKER_03", "SPEAKER_04"]
        idx = 0
        for i, seg in enumerate(segments):
            if i > 0 and (seg["start"] - segments[i-1]["end"]) >= 1.2:
                idx = (idx + 1) % len(SPEAKERS)
            seg["speaker"] = SPEAKERS[idx]
        return segments

    def _build_speaker_data(self, segments: List[Dict]) -> Dict:
        data = {}
        for seg in segments:
            spk = seg["speaker"]
            if spk not in data:
                data[spk] = {"text": "", "segments": [], "total_duration": 0.0,
                              "segment_count": 0, "first_time": seg["start"],
                              "last_time": seg["end"]}
            d = data[spk]
            d["text"] += (" " + seg["text"]).lstrip()
            d["segments"].append({"start": seg["start"], "end": seg["end"], "text": seg["text"]})
            d["total_duration"] += round(seg["end"] - seg["start"], 2)
            d["segment_count"]  += 1
            d["last_time"]       = max(d["last_time"], seg["end"])
        for d in data.values():
            d["text"] = d["text"].strip()
        return dict(sorted(data.items(), key=lambda x: x[1]["first_time"]))


# ─────────────────────────────────────────────────────────────────────────────
# Summarizer
# ─────────────────────────────────────────────────────────────────────────────
class Summarizer:
    def ollama(self, text: str, model: str, max_w: int = 180, extra: str = "") -> Tuple[str, str]:
        import ollama
        prompt = (f"Summarise the following in {max_w} words or fewer. "
                  f"Be concise, accurate, cover key points. {extra}\n\nTRANSCRIPT:\n{text[:4000]}\n\nSUMMARY:")
        r = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        return r["message"]["content"].strip(), model

    def bart(self, text: str) -> Tuple[str, str]:
        try:
            from transformers import pipeline
            p = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
            o = p(text[:3500], max_length=220, min_length=60, do_sample=False)
            return o[0]["summary_text"].strip(), "BART (distilbart-cnn-12-6)"
        except Exception:
            return self._extractive(text, 5), "Extractive TF-IDF"

    def _extractive(self, text: str, n: int = 4) -> str:
        STOP = {"the","a","an","in","on","at","to","for","of","and","or","but","is","are",
                "was","were","be","been","it","this","that","with","i","we","you","he",
                "she","they","so","as","by","from","not","have","has","had","do","did",
                "will","would","could","should"}
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 6]
        if not sents: return text[:400]
        freq = Counter(re.findall(r"\b\w+\b", text.lower()))
        scored = []
        for s in sents:
            words = [w for w in re.findall(r"\b\w+\b", s.lower()) if w not in STOP]
            score = sum(freq[w] for w in words) / (len(words)+1)
            scored.append((score, s))
        scored.sort(reverse=True)
        top = sorted(scored[:n], key=lambda x: sents.index(x[1]))
        return " ".join(s[1] for s in top)

    def speaker_summary(self, text: str, spk: str, model: str) -> str:
        if len(text.split()) < 20:
            return text.strip()
        try:
            s, _ = self.ollama(text, model, 80, f"Focus on what {spk_label(spk)} said.")
            return s
        except Exception:
            return self._extractive(text, 2)

    def pick_best(self, sum_a, name_a, sum_b, name_b, full_text,
                  judge_model=None) -> Tuple[str, str, str]:
        a_ok = not str(sum_a).startswith("⚠")
        b_ok = not str(sum_b).startswith("⚠")
        if not a_ok and not b_ok: return "No summary available.", "—", "Both failed."
        if not a_ok: return sum_b, f"✅ {name_b}", "Only available summary."
        if not b_ok: return sum_a, f"✅ {name_a}", "Only available summary."
        if judge_model:
            try:
                import ollama
                prompt = (
                    "You are a judge. Which summary is more accurate and concise?\n"
                    "Reply ONLY: A: <reason>  OR  B: <reason>\n\n"
                    f"SUMMARY A:\n{sum_a}\n\nSUMMARY B:\n{sum_b}\n\nVerdict:"
                )
                r = ollama.chat(model=judge_model, messages=[{"role":"user","content":prompt}])
                ans = r["message"]["content"].strip()
                if ans.upper().startswith("A"):
                    return sum_a, f"✅ {name_a}", ans[2:].strip() or "Preferred by Ollama"
                return sum_b, f"✅ {name_b}", ans[2:].strip() or "Preferred by Ollama"
            except Exception:
                pass
        sa = self._quality(sum_a, full_text)
        sb = self._quality(sum_b, full_text)
        if sa >= sb: return sum_a, f"✅ {name_a}", f"Quality score {sa:.3f} vs {sb:.3f}"
        return sum_b, f"✅ {name_b}", f"Quality score {sb:.3f} vs {sa:.3f}"

    def _quality(self, summary: str, full_text: str) -> float:
        wc = len(summary.split())
        l = 1.0 if 40 <= wc <= 280 else (wc/40 if wc<40 else max(.3, 1-(wc-280)/280))
        fw = set(re.findall(r"\b\w{4,}\b", full_text.lower()))
        sw = set(re.findall(r"\b\w{4,}\b", summary.lower()))
        return round(l * len(fw & sw) / (len(fw)+1), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Excel Export
# ─────────────────────────────────────────────────────────────────────────────
def export_excel(session: Dict, out_path: str):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    ws = wb.active
    ws.title = "Full Results"

    def fill(hex_c): return PatternFill("solid", fgColor=hex_c.lstrip("#"))
    def border():
        s = Side(style="thin", color="C0C0C0")
        return Border(left=s, right=s, top=s, bottom=s)
    def hdr(ws, row, col, val, bg, fg="FFFFFF", sz=11, bold=True):
        c = ws.cell(row=row, column=col, value=val)
        c.font = Font(name="Calibri", bold=bold, size=sz, color=fg)
        c.fill = fill(bg); c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = border(); return c
    def cell(ws, row, col, val, bg="FFFFFF", fg="111111", sz=10, italic=False, bold=False, wrap=True):
        c = ws.cell(row=row, column=col, value=val)
        c.font = Font(name="Calibri", bold=bold, italic=italic, size=sz, color=fg)
        c.fill = fill(bg); c.alignment = Alignment(vertical="top", wrap_text=wrap)
        c.border = border(); return c

    # ── Sheet 1: Full Results ──────────────────────────────────────────────
    col_w = [15, 13, 52, 42, 42, 42]
    for i, w in enumerate(col_w, 1):
        ws.column_dimensions[get_column_letter(i)].width = w

    ws.merge_cells("A1:F1")
    t = ws["A1"]
    t.value     = f"SpeakSense Recording — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
    t.font      = Font(name="Calibri", bold=True, size=15, color="FFFFFF")
    t.fill      = fill("0D1B2A"); t.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 34

    HDRS = ["TIMELINE","SPEAKER","WHAT WAS SPOKEN","SPEAKER SUMMARY (AI)","OTHER AI SUMMARY","✅ BEST SUMMARY"]
    HCS  = ["1B4F72","1E8449","4A235A","784212","117A65","145A32"]
    for c, (h, hc) in enumerate(zip(HDRS, HCS), 1):
        hdr(ws, 2, c, h, hc)
    ws.row_dimensions[2].height = 26

    segments      = session.get("segments", [])
    spk_summaries = session.get("speaker_summaries", {})
    alt_sum       = session.get("alt_ai_summary", "")
    best_sum      = session.get("best_summary", "")

    SPK_LIGHT = {"SPEAKER_01":"D6EAF8","SPEAKER_02":"D5F5E3","SPEAKER_03":"FDEBD0",
                 "SPEAKER_04":"F5EEF8","SPEAKER_05":"D1F2EB","SPEAKER_06":"D6EAF8"}
    SPK_DARK  = {"SPEAKER_01":"1A5276","SPEAKER_02":"1E8449","SPEAKER_03":"784212",
                 "SPEAKER_04":"76448A","SPEAKER_05":"117A65","SPEAKER_06":"1A5276"}

    seen = set()
    for ri, seg in enumerate(segments, 3):
        spk   = seg.get("speaker","SPEAKER_01")
        light = SPK_LIGHT.get(spk,"EEF2FF")
        dark  = SPK_DARK.get(spk,"1A1A1A")
        is_first = spk not in seen; seen.add(spk)
        tl = f"{fmt_time(seg['start'])} → {fmt_time(seg['end'])}"
        cell(ws, ri, 1, tl, light, "555555"); ws.cell(ri,1).font = Font(name="Courier New", size=9, color="555555")
        c2 = cell(ws, ri, 2, spkLabel(spk), dark, "FFFFFF", bold=True)
        c2.alignment = Alignment(horizontal="center", vertical="top")
        cell(ws, ri, 3, seg.get("text",""), light)
        cell(ws, ri, 4, spk_summaries.get(spk,"") if is_first else "", "EBF5FB", italic=True, sz=9)
        cell(ws, ri, 5, alt_sum if ri==3 else "", "F9EBEA", "4A235A", italic=True, sz=9)
        c6 = cell(ws, ri, 6, best_sum if ri==3 else "", "EAFAF1", "145A32", bold=(ri==3), sz=9)
        h = max(18, min(72, len(seg.get("text",""))//3+18))
        ws.row_dimensions[ri].height = h

    ws.freeze_panes = "A3"

    # ── Sheet 2: AI Comparison ─────────────────────────────────────────────
    ws2 = wb.create_sheet("AI Comparison")
    ws2.column_dimensions["A"].width = 22
    ws2.column_dimensions["B"].width = 90
    ws2.merge_cells("A1:B1")
    t2 = ws2["A1"]; t2.value = "AI Summary Comparison"
    t2.font = Font(name="Calibri", bold=True, size=14, color="FFFFFF")
    t2.fill = fill("1A252F"); t2.alignment = Alignment(horizontal="center", vertical="center")
    ws2.row_dimensions[1].height = 30
    rows2 = [
        ("Ollama\n"+session.get("ollama_model_used",""),   session.get("ollama_summary",""),  "1A5276","D6EAF8"),
        ("Alt AI\n"+session.get("alt_ai_name",""),         alt_sum,                           "4A235A","F5EEF8"),
        ("✅ Best\n"+session.get("best_summary_source",""), best_sum,                          "145A32","EAFAF1"),
        ("Selection Reason",                               session.get("best_summary_reason",""),"1B2631","FDFEFE"),
    ]
    for ri, (lbl, val, hc, vc) in enumerate(rows2, 2):
        c1 = ws2.cell(row=ri, column=1, value=lbl)
        c1.font=Font(name="Calibri",bold=True,size=11,color="FFFFFF"); c1.fill=fill(hc)
        c1.alignment=Alignment(vertical="top",wrap_text=True); c1.border=border()
        c2 = ws2.cell(row=ri, column=2, value=val)
        c2.font=Font(name="Calibri",size=11); c2.fill=fill(vc)
        c2.alignment=Alignment(horizontal="left",vertical="top",wrap_text=True); c2.border=border()
        ws2.row_dimensions[ri].height = max(28, min(120, len(str(val))//2+18))

    # ── Sheet 3: Speaker Stats ─────────────────────────────────────────────
    ws3 = wb.create_sheet("Speaker Stats")
    for i,w in enumerate([14,14,12,12,14,12,80],1): ws3.column_dimensions[get_column_letter(i)].width=w
    ws3.merge_cells("A1:G1"); t3=ws3["A1"]; t3.value="Speaker Statistics"
    t3.font=Font(name="Calibri",bold=True,size=14,color="FFFFFF")
    t3.fill=fill("212F3D"); t3.alignment=Alignment(horizontal="center",vertical="center")
    ws3.row_dimensions[1].height=28
    SHDR=["SPEAKER","LABEL","FIRST","LAST","DURATION (s)","SEGMENTS","FULL TEXT"]
    for c,h in enumerate(SHDR,1): hdr(ws3,2,c,h,"34495E")
    spk_data = session.get("speaker_data",{})
    for ri,(spk,info) in enumerate(spk_data.items(),3):
        dark=SPK_DARK.get(spk,"1A1A1A"); light=SPK_LIGHT.get(spk,"FFFFFF")
        vals=[spk, spkLabel(spk), fmt_time(info.get("first_time",0)),
              fmt_time(info.get("last_time",0)), round(info.get("total_duration",0),1),
              info.get("segment_count",0), info.get("text","")]
        for ci,v in enumerate(vals,1):
            c=ws3.cell(row=ri,column=ci,value=v)
            c.font=Font(name="Calibri",size=10,bold=ci<=2,color="FFFFFF" if ci<=2 else "111111")
            c.fill=fill(dark if ci<=2 else light)
            c.alignment=Alignment(vertical="top",wrap_text=(ci==7)); c.border=border()
    ws3.freeze_panes="A3"

    wb.save(out_path)

def spkLabel(spk): return f"Speaker {int(spk.replace('SPEAKER_',''),10)}"


# ─────────────────────────────────────────────────────────────────────────────
# Main Application Window
# ─────────────────────────────────────────────────────────────────────────────
class SpeakSenseApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("SpeakSense — Real-Time Speech Summarizer")
        self.geometry("1440x900")
        self.minsize(1100, 700)
        self.configure(fg_color=PALETTE["bg"])

        # State
        self._is_recording   = False
        self._audio_frames   = []
        self._audio_queue    = queue.Queue()
        self._session_data   = {}
        self._timer_seconds  = 0
        self._timer_job      = None
        self._processor      = None
        self._summarizer     = Summarizer()
        self._waveform_data  = np.zeros(200)
        self._waveform_job   = None

        self._build_ui()
        self._start_waveform_loop()

    # ──────────────────────────────────────────────────────────────────────
    # UI Construction
    # ──────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Top bar ────────────────────────────────────────────────────────
        top = ctk.CTkFrame(self, fg_color=PALETTE["surface"], height=58, corner_radius=0)
        top.pack(fill="x", side="top")
        top.pack_propagate(False)

        ctk.CTkLabel(top, text="🎙  SpeakSense",
                     font=ctk.CTkFont("Segoe UI", 20, "bold"),
                     text_color=PALETTE["blue"]).pack(side="left", padx=20, pady=12)

        # Status badge
        self._status_frame = ctk.CTkFrame(top, fg_color=PALETTE["card"],
                                           corner_radius=20, height=30)
        self._status_frame.pack(side="right", padx=16, pady=14)
        self._status_dot   = ctk.CTkLabel(self._status_frame, text="●",
                                           font=ctk.CTkFont(size=12),
                                           text_color=PALETTE["muted"], width=18)
        self._status_dot.pack(side="left", padx=(10,2), pady=4)
        self._status_label = ctk.CTkLabel(self._status_frame, text="READY",
                                           font=ctk.CTkFont("Courier", 11, "bold"),
                                           text_color=PALETTE["muted"], width=110)
        self._status_label.pack(side="left", padx=(0,12), pady=4)

        # Timer
        self._timer_label = ctk.CTkLabel(top, text="00:00",
                                          font=ctk.CTkFont("Courier", 22, "bold"),
                                          text_color=PALETTE["text"])
        self._timer_label.pack(side="right", padx=10)

        # ── Controls row ───────────────────────────────────────────────────
        ctrl = ctk.CTkFrame(self, fg_color=PALETTE["card"], corner_radius=0, height=70)
        ctrl.pack(fill="x")
        ctrl.pack_propagate(False)

        # Whisper model
        ctk.CTkLabel(ctrl, text="WHISPER MODEL",
                     font=ctk.CTkFont("Courier", 9), text_color=PALETTE["muted"]).pack(side="left", padx=(18,4), pady=(10,0))
        self._whisper_var = ctk.StringVar(value="base")
        ctk.CTkOptionMenu(ctrl, variable=self._whisper_var,
                          values=["tiny","base","small","medium","large"],
                          width=130, height=32,
                          fg_color=PALETTE["surface"], button_color=PALETTE["raised"],
                          font=ctk.CTkFont("Courier", 12)).pack(side="left", padx=(0,18), pady=18)

        # Ollama model
        ctk.CTkLabel(ctrl, text="OLLAMA MODEL",
                     font=ctk.CTkFont("Courier", 9), text_color=PALETTE["muted"]).pack(side="left", padx=(0,4), pady=(10,0))
        self._ollama_var = ctk.StringVar(value="llama3")
        ctk.CTkOptionMenu(ctrl, variable=self._ollama_var,
                          values=["llama3","llama3.1","mistral","gemma2","phi3","qwen2"],
                          width=140, height=32,
                          fg_color=PALETTE["surface"], button_color=PALETTE["raised"],
                          font=ctk.CTkFont("Courier", 12)).pack(side="left", padx=(0,28), pady=18)

        # Buttons
        self._btn_start = ctk.CTkButton(ctrl, text="⏺  Start Recording",
                                         width=180, height=36,
                                         fg_color=PALETTE["red"],
                                         hover_color="#cc2f3d",
                                         font=ctk.CTkFont("Segoe UI", 13, "bold"),
                                         command=self._start_recording)
        self._btn_start.pack(side="left", padx=4, pady=16)

        self._btn_stop = ctk.CTkButton(ctrl, text="⏹  Stop & Analyze",
                                        width=180, height=36, state="disabled",
                                        fg_color=PALETTE["raised"],
                                        hover_color=PALETTE["border"],
                                        font=ctk.CTkFont("Segoe UI", 13, "bold"),
                                        command=self._stop_recording)
        self._btn_stop.pack(side="left", padx=4, pady=16)

        self._btn_excel = ctk.CTkButton(ctrl, text="📊  Export to Excel",
                                         width=180, height=36, state="disabled",
                                         fg_color=PALETTE["green"],
                                         hover_color="#1dba60",
                                         text_color="#051a0d",
                                         font=ctk.CTkFont("Segoe UI", 13, "bold"),
                                         command=self._export_excel)
        self._btn_excel.pack(side="left", padx=4, pady=16)

        # ── Waveform canvas ────────────────────────────────────────────────
        self._wave_frame = ctk.CTkFrame(self, fg_color=PALETTE["card"],
                                         corner_radius=0, height=70)
        self._wave_frame.pack(fill="x")
        self._wave_frame.pack_propagate(False)

        self._wave_canvas = tk.Canvas(self._wave_frame, bg=PALETTE["card"],
                                       highlightthickness=0, height=70)
        self._wave_canvas.pack(fill="both", expand=True, padx=16, pady=8)

        # Progress bar (hidden by default)
        self._progress_frame = ctk.CTkFrame(self, fg_color=PALETTE["surface"],
                                             corner_radius=0, height=40)
        self._progress_label = ctk.CTkLabel(self._progress_frame, text="",
                                             font=ctk.CTkFont("Courier", 11),
                                             text_color=PALETTE["blue"])
        self._progress_label.pack(side="left", padx=16, pady=10)
        self._progress_bar   = ctk.CTkProgressBar(self._progress_frame, width=280,
                                                   fg_color=PALETTE["border"],
                                                   progress_color=PALETTE["blue"])
        self._progress_bar.pack(side="left", padx=10, pady=10)
        self._progress_bar.set(0)

        # ── Main body ──────────────────────────────────────────────────────
        body = ctk.CTkFrame(self, fg_color=PALETTE["bg"], corner_radius=0)
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(1, weight=1)

        # Timeline section
        tl_hdr = ctk.CTkFrame(body, fg_color=PALETTE["bg"], height=30)
        tl_hdr.grid(row=0, column=0, sticky="ew", padx=16, pady=(12,4))
        ctk.CTkLabel(tl_hdr, text="🕒  SPEAKER TIMELINE",
                     font=ctk.CTkFont("Courier", 10, "bold"),
                     text_color=PALETTE["muted"]).pack(side="left")

        self._tl_frame = ctk.CTkFrame(body, fg_color=PALETTE["card"],
                                       corner_radius=12, height=120)
        self._tl_frame.grid(row=1, column=0, sticky="ew", padx=16, pady=(0,8))
        self._tl_frame.grid_propagate(False)
        self._tl_canvas = tk.Canvas(self._tl_frame, bg=PALETTE["card"],
                                     highlightthickness=0, height=120)
        self._tl_canvas.pack(fill="both", expand=True, padx=12, pady=8)
        self._tl_canvas.bind("<Configure>", lambda e: self._render_timeline())

        # ── Notebook tabs ──────────────────────────────────────────────────
        tab_frame = ctk.CTkFrame(body, fg_color=PALETTE["bg"], corner_radius=0)
        tab_frame.grid(row=2, column=0, sticky="nsew", padx=16, pady=(0,10))
        body.grid_rowconfigure(2, weight=1)

        self._notebook = ctk.CTkTabview(tab_frame,
                                         fg_color=PALETTE["card"],
                                         segmented_button_fg_color=PALETTE["surface"],
                                         segmented_button_selected_color=PALETTE["blue"],
                                         segmented_button_unselected_color=PALETTE["raised"],
                                         segmented_button_selected_hover_color="#2a7fd8",
                                         text_color=PALETTE["text"])
        self._notebook.pack(fill="both", expand=True)

        self._notebook.add("📋  Full Transcript")
        self._notebook.add("👤  Speaker Profiles")
        self._notebook.add("🤖  AI Comparison")

        self._build_transcript_tab()
        self._build_speakers_tab()
        self._build_ai_tab()

    # ──────────────────────────────────────────────────────────────────────
    # Tab: Full Transcript Table
    # ──────────────────────────────────────────────────────────────────────
    def _build_transcript_tab(self):
        tab = self._notebook.tab("📋  Full Transcript")
        tab.configure(fg_color=PALETTE["card"])

        cols = ("timeline", "speaker", "spoken", "spk_summary", "alt_ai", "best")
        hdrs = ("TIMELINE", "SPEAKER", "WHAT WAS SPOKEN",
                "SPEAKER SUMMARY", "OTHER AI SUMMARY", "✅ BEST SUMMARY")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.Treeview",
                         background=PALETTE["card"], foreground=PALETTE["text"],
                         rowheight=54, borderwidth=0, relief="flat",
                         font=("Segoe UI", 11))
        style.configure("Dark.Treeview.Heading",
                         background=PALETTE["raised"], foreground=PALETTE["muted"],
                         font=("Courier", 9, "bold"), relief="flat",
                         borderwidth=1)
        style.map("Dark.Treeview", background=[("selected","#1c3a5e")],
                                    foreground=[("selected", PALETTE["text"])])
        style.configure("Vertical.TScrollbar",
                         troughcolor=PALETTE["surface"], background=PALETTE["border"])

        frame = ctk.CTkFrame(tab, fg_color=PALETTE["card"], corner_radius=0)
        frame.pack(fill="both", expand=True, padx=8, pady=8)

        tree = ttk.Treeview(frame, columns=cols, show="headings",
                             style="Dark.Treeview", selectmode="browse")

        col_widths = [110, 90, 280, 220, 220, 220]
        for c, h, w in zip(cols, hdrs, col_widths):
            tree.heading(c, text=h)
            tree.column(c, width=w, minwidth=80, anchor="w")

        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        # Tag colors per speaker
        for spk, clr in SPEAKER_COLORS.items():
            tree.tag_configure(spk, foreground=clr)
        tree.tag_configure("best_row", background="#0d2b1e")

        self._tree = tree

    # ──────────────────────────────────────────────────────────────────────
    # Tab: Speaker Profiles
    # ──────────────────────────────────────────────────────────────────────
    def _build_speakers_tab(self):
        tab = self._notebook.tab("👤  Speaker Profiles")
        tab.configure(fg_color=PALETTE["card"])
        scroll = ctk.CTkScrollableFrame(tab, fg_color=PALETTE["card"], corner_radius=0)
        scroll.pack(fill="both", expand=True, padx=8, pady=8)
        self._spk_scroll = scroll
        self._spk_widgets = {}  # spk → {frame, name_lbl, stats_lbl, sum_lbl}

    def _rebuild_speaker_cards(self):
        # Clear old
        for w in self._spk_scroll.winfo_children():
            w.destroy()
        self._spk_widgets.clear()

        cols = 3
        for i, (spk, info) in enumerate(self._session_data.get("speaker_data",{}).items()):
            clr = spk_color(spk)
            bg  = SPEAKER_BG.get(spk, PALETTE["card"])
            card = ctk.CTkFrame(self._spk_scroll, fg_color=bg,
                                corner_radius=12, border_width=1,
                                border_color=clr+"55")
            card.grid(row=i//cols, column=i%cols, padx=8, pady=8, sticky="nsew")
            self._spk_scroll.grid_columnconfigure(i%cols, weight=1)

            hdr = ctk.CTkFrame(card, fg_color=clr+"22", corner_radius=8)
            hdr.pack(fill="x", padx=10, pady=(10,6))
            ctk.CTkLabel(hdr, text=f"  S{int(spk.replace('SPEAKER_',''),10)}",
                         font=ctk.CTkFont("Segoe UI", 18, "bold"),
                         text_color=clr).pack(side="left", padx=8, pady=8)
            name_f = ctk.CTkFrame(hdr, fg_color="transparent")
            name_f.pack(side="left", pady=8)
            nl = ctk.CTkLabel(name_f, text=spk_label(spk),
                               font=ctk.CTkFont("Segoe UI", 13, "bold"), text_color=clr)
            nl.pack(anchor="w")
            sl = ctk.CTkLabel(name_f,
                               text=f"{info['segment_count']} segments · {info['total_duration']:.1f}s",
                               font=ctk.CTkFont("Courier", 9), text_color=PALETTE["muted"])
            sl.pack(anchor="w")

            ctk.CTkLabel(card, text="AI SUMMARY",
                          font=ctk.CTkFont("Courier", 9, "bold"),
                          text_color=PALETTE["muted"]).pack(anchor="w", padx=14, pady=(6,2))
            sum_lbl = ctk.CTkLabel(card, text="Generating…",
                                    font=ctk.CTkFont("Segoe UI", 11),
                                    text_color=PALETTE["text"],
                                    wraplength=300, justify="left")
            sum_lbl.pack(anchor="w", padx=14, pady=(0,12))

            self._spk_widgets[spk] = sum_lbl

    def _update_speaker_card(self, spk: str, summary: str):
        if spk in self._spk_widgets:
            self._spk_widgets[spk].configure(text=summary)

    # ──────────────────────────────────────────────────────────────────────
    # Tab: AI Comparison
    # ──────────────────────────────────────────────────────────────────────
    def _build_ai_tab(self):
        tab = self._notebook.tab("🤖  AI Comparison")
        tab.configure(fg_color=PALETTE["card"])
        tab.grid_columnconfigure((0,1,2), weight=1)
        tab.grid_rowconfigure(1, weight=1)

        specs = [
            ("🦙  Ollama Summary",    "ai_ollama_model", "ai_ollama_text",  PALETTE["purple"]),
            ("🤗  Alt AI Summary",    "ai_alt_model",    "ai_alt_text",     PALETTE["amber"]),
            ("✅  Best Summary",      "ai_best_source",  "ai_best_text",    PALETTE["green"]),
        ]
        for col, (title, model_attr, text_attr, color) in enumerate(specs):
            box = ctk.CTkFrame(tab, fg_color=PALETTE["raised"],
                               corner_radius=12, border_width=1,
                               border_color=color+"44")
            box.grid(row=0, column=col, padx=8, pady=8, sticky="nsew")
            tab.grid_rowconfigure(0, weight=1)

            ctk.CTkLabel(box, text=title,
                          font=ctk.CTkFont("Segoe UI", 13, "bold"),
                          text_color=color).pack(anchor="w", padx=14, pady=(12,2))
            model_lbl = ctk.CTkLabel(box, text="—",
                                      font=ctk.CTkFont("Courier", 9),
                                      text_color=PALETTE["muted"])
            model_lbl.pack(anchor="w", padx=14, pady=(0,8))
            sep = ctk.CTkFrame(box, fg_color=PALETTE["border"], height=1)
            sep.pack(fill="x", padx=14, pady=(0,10))
            text_lbl = ctk.CTkLabel(box, text="—",
                                     font=ctk.CTkFont("Segoe UI", 11),
                                     text_color=PALETTE["text"],
                                     wraplength=380, justify="left")
            text_lbl.pack(anchor="nw", padx=14, pady=(0,12), fill="both", expand=True)
            setattr(self, model_attr, model_lbl)
            setattr(self, text_attr,  text_lbl)

        # Reason row (below)
        reason_box = ctk.CTkFrame(tab, fg_color=PALETTE["surface"], corner_radius=10)
        reason_box.grid(row=1, column=0, columnspan=3, padx=8, pady=(0,8), sticky="ew")
        ctk.CTkLabel(reason_box, text="SELECTION REASON",
                      font=ctk.CTkFont("Courier", 9, "bold"),
                      text_color=PALETTE["muted"]).pack(side="left", padx=14, pady=10)
        self._reason_lbl = ctk.CTkLabel(reason_box, text="—",
                                         font=ctk.CTkFont("Segoe UI", 11, "italic"),
                                         text_color=PALETTE["muted"], wraplength=900)
        self._reason_lbl.pack(side="left", padx=8, pady=10)

    # ──────────────────────────────────────────────────────────────────────
    # Recording
    # ──────────────────────────────────────────────────────────────────────
    def _start_recording(self):
        import sounddevice as sd

        self._audio_frames.clear()
        self._is_recording = True
        self._timer_seconds = 0
        self._btn_start.configure(state="disabled", fg_color="#7a0f18")
        self._btn_stop.configure(state="normal")
        self._btn_excel.configure(state="disabled")
        self._set_status("●  RECORDING", PALETTE["red"])
        self._clear_results()
        self._start_timer()

        sr = 16000
        self._sd_stream = sd.InputStream(
            samplerate=sr, channels=1, dtype="int16",
            blocksize=4096, callback=self._audio_callback)
        self._sd_stream.start()

    def _audio_callback(self, indata, frames, time_info, status):
        if self._is_recording:
            self._audio_frames.append(indata.copy())
            chunk = indata[:, 0].astype(np.float32) / 32768.0
            level = np.abs(chunk).mean()
            self._waveform_data = np.roll(self._waveform_data, -1)
            self._waveform_data[-1] = level

    def _stop_recording(self):
        if not self._is_recording:
            return
        self._is_recording = False
        self._btn_stop.configure(state="disabled")
        self._btn_start.configure(state="disabled")
        self._stop_timer()

        if hasattr(self, "_sd_stream"):
            self._sd_stream.stop()
            self._sd_stream.close()

        if not self._audio_frames:
            mb.showerror("No audio", "No audio was captured. Check your microphone.")
            self._btn_start.configure(state="normal")
            return

        # Save WAV
        wav_path = str(Path("recordings") / f"{uuid.uuid4().hex[:8]}.wav")
        Path("recordings").mkdir(exist_ok=True)
        self._save_wav(wav_path)

        # Process in background
        self._set_status("⚙  PROCESSING", PALETTE["amber"])
        self._show_progress(True)
        threading.Thread(target=self._process_pipeline,
                         args=(wav_path,), daemon=True).start()

    def _save_wav(self, path: str):
        audio = np.concatenate(self._audio_frames, axis=0)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio.tobytes())

    # ──────────────────────────────────────────────────────────────────────
    # Pipeline
    # ──────────────────────────────────────────────────────────────────────
    def _process_pipeline(self, wav_path: str):
        try:
            whisper_model = self._whisper_var.get()
            ollama_model  = self._ollama_var.get()

            # ── Load Whisper ──────────────────────────────────────────────
            self._emit("progress", (0.1, "🎙 Loading Whisper model…"))
            if self._processor is None or self._processor.model_size != whisper_model:
                self._processor = AudioProcessor(whisper_model)

            # ── Transcribe + Diarize ──────────────────────────────────────
            self._emit("progress", (0.25, "🎙 Transcribing audio…"))
            data = self._processor.transcribe_and_diarize(wav_path)
            self._session_data.update(data)
            self._emit("transcript_ready", data)

            # ── Per-speaker summaries ─────────────────────────────────────
            self._emit("progress", (0.50, "📝 Summarizing speakers…"))
            spk_summaries = {}
            for spk, info in data["speaker_data"].items():
                s = self._summarizer.speaker_summary(info["text"], spk, ollama_model)
                spk_summaries[spk] = s
                self._emit("speaker_summary", (spk, s))

            # ── Ollama full summary ───────────────────────────────────────
            self._emit("progress", (0.68, "🦙 Running Ollama summary…"))
            try:
                ollama_sum, ollama_model_used = self._summarizer.ollama(data["full_text"], ollama_model)
            except Exception as e:
                ollama_sum, ollama_model_used = f"⚠ {e}", "N/A"

            # ── Alt AI summary ────────────────────────────────────────────
            self._emit("progress", (0.82, "📚 Running BART / Alt AI summary…"))
            alt_sum, alt_name = self._summarizer.bart(data["full_text"])

            # ── Judge ─────────────────────────────────────────────────────
            self._emit("progress", (0.93, "⚖️ Picking best summary…"))
            best_sum, best_src, best_reason = self._summarizer.pick_best(
                ollama_sum, f"Ollama ({ollama_model})",
                alt_sum,   alt_name,
                data["full_text"],
                judge_model=ollama_model if not ollama_sum.startswith("⚠") else None
            )

            self._session_data.update({
                "speaker_summaries":   spk_summaries,
                "ollama_summary":      ollama_sum,
                "ollama_model_used":   ollama_model_used,
                "alt_ai_summary":      alt_sum,
                "alt_ai_name":         alt_name,
                "best_summary":        best_sum,
                "best_summary_source": best_src,
                "best_summary_reason": best_reason,
            })

            self._emit("complete", {
                "spk_summaries":       spk_summaries,
                "ollama_summary":      ollama_sum,
                "ollama_model_used":   ollama_model_used,
                "alt_ai_summary":      alt_sum,
                "alt_ai_name":         alt_name,
                "best_summary":        best_sum,
                "best_summary_source": best_src,
                "best_summary_reason": best_reason,
            })

        except Exception as e:
            import traceback; traceback.print_exc()
            self._emit("error", str(e))

    # ──────────────────────────────────────────────────────────────────────
    # Event dispatch (thread-safe via after())
    # ──────────────────────────────────────────────────────────────────────
    def _emit(self, event: str, data):
        self.after(0, lambda: self._handle_event(event, data))

    def _handle_event(self, event: str, data):
        if event == "progress":
            pct, label = data
            self._progress_bar.set(pct)
            self._progress_label.configure(text=label)

        elif event == "transcript_ready":
            self._render_timeline()
            self._fill_table_rows()
            self._rebuild_speaker_cards()

        elif event == "speaker_summary":
            spk, summary = data
            self._update_speaker_card(spk, summary)
            self._update_table_speaker_summary(spk, summary)

        elif event == "complete":
            d = data
            self.ai_ollama_model.configure(text=d["ollama_model_used"])
            self.ai_alt_model.configure(text=d["alt_ai_name"])
            self.ai_best_source.configure(text=d["best_summary_source"])
            self.ai_ollama_text.configure(text=d["ollama_summary"])
            self.ai_alt_text.configure(text=d["alt_ai_summary"])
            self.ai_best_text.configure(text=d["best_summary"])
            self._reason_lbl.configure(text=d["best_summary_reason"])
            self._fill_alt_best_cols(d["alt_ai_summary"], d["best_summary"])
            self._show_progress(False)
            self._btn_start.configure(state="normal", fg_color=PALETTE["red"])
            self._btn_excel.configure(state="normal")
            self._set_status("✅  COMPLETE", PALETTE["green"])

        elif event == "error":
            self._show_progress(False)
            self._btn_start.configure(state="normal", fg_color=PALETTE["red"])
            self._set_status("✗  ERROR", PALETTE["red"])
            mb.showerror("Processing Error", str(data))

    # ──────────────────────────────────────────────────────────────────────
    # Table population
    # ──────────────────────────────────────────────────────────────────────
    def _fill_table_rows(self):
        self._tree.delete(*self._tree.get_children())
        segments = self._session_data.get("segments", [])
        self._row_spk_map = {}  # iid → spk

        seen = set()
        for seg in segments:
            spk = seg.get("speaker","SPEAKER_01")
            tl  = f"{fmt_time(seg['start'])} → {fmt_time(seg['end'])}"
            lbl = spk_label(spk)
            iid = self._tree.insert("", "end",
                                    values=(tl, lbl, seg["text"], "", "", ""),
                                    tags=(spk,))
            self._row_spk_map[iid] = spk
            seen.add(spk)

    def _update_table_speaker_summary(self, spk: str, summary: str):
        first = True
        for iid in self._tree.get_children():
            if self._row_spk_map.get(iid) == spk and first:
                vals = list(self._tree.item(iid)["values"])
                vals[3] = summary
                self._tree.item(iid, values=vals)
                first = False

    def _fill_alt_best_cols(self, alt: str, best: str):
        children = self._tree.get_children()
        if children:
            iid  = children[0]
            vals = list(self._tree.item(iid)["values"])
            vals[4] = alt; vals[5] = best
            self._tree.item(iid, values=vals)
            self._tree.item(iid, tags=(self._row_spk_map.get(iid,"SPEAKER_01"), "best_row"))

    # ──────────────────────────────────────────────────────────────────────
    # Timeline
    # ──────────────────────────────────────────────────────────────────────
    def _render_timeline(self):
        c = self._tl_canvas
        c.delete("all")
        data     = self._session_data.get("speaker_data", {})
        duration = self._session_data.get("duration", 0)
        if not data or not duration:
            c.create_text(20, 60, text="No timeline data yet",
                          fill=PALETTE["muted"], anchor="w",
                          font=("Courier", 11))
            return

        W = c.winfo_width() or 900
        H = c.winfo_height() or 120
        LABEL_W = 90
        TRACK_W = W - LABEL_W - 20
        LANE_H  = 18
        LANE_PAD = 6
        TOP     = 10

        for i, (spk, info) in enumerate(data.items()):
            clr = spk_color(spk)
            y0  = TOP + i * (LANE_H + LANE_PAD)
            # Background lane
            c.create_rectangle(LABEL_W, y0, LABEL_W + TRACK_W, y0 + LANE_H,
                                fill=PALETTE["surface"], outline="")
            # Label
            c.create_text(LABEL_W - 6, y0 + LANE_H//2, text=spk_label(spk),
                           fill=clr, anchor="e",
                           font=("Courier", 9, "bold"))
            # Segments
            for seg in info.get("segments", []):
                x1 = LABEL_W + (seg["start"] / duration) * TRACK_W
                x2 = LABEL_W + (seg["end"]   / duration) * TRACK_W
                x2 = max(x1 + 3, x2)
                c.create_rectangle(x1, y0+1, x2, y0+LANE_H-1,
                                   fill=clr, outline="")

        # Time ticks
        ticks = min(10, int(duration / 30) + 2)
        for i in range(ticks+1):
            t = (i / ticks) * duration
            x = LABEL_W + (t / duration) * TRACK_W
            y_bot = TOP + len(data) * (LANE_H + LANE_PAD) + 4
            c.create_line(x, TOP, x, y_bot, fill=PALETTE["border"], dash=(2,4))
            c.create_text(x, y_bot + 8, text=fmt_time(t),
                          fill=PALETTE["muted"], font=("Courier", 8), anchor="n")

    # ──────────────────────────────────────────────────────────────────────
    # Waveform
    # ──────────────────────────────────────────────────────────────────────
    def _start_waveform_loop(self):
        self._draw_waveform()

    def _draw_waveform(self):
        try:
            c = self._wave_canvas
            W = c.winfo_width()  or 900
            H = c.winfo_height() or 70
            c.delete("all")
            if self._is_recording:
                n    = len(self._waveform_data)
                bw   = W / n
                for i, v in enumerate(self._waveform_data):
                    bh = max(2, v * H * 4)
                    x  = i * bw
                    alpha_hex = format(int(80 + v * 175), '02x')
                    color = f"#ff{alpha_hex}57" if v > 0.05 else PALETTE["border"]
                    c.create_rectangle(x, H//2-bh//2, x+bw-1, H//2+bh//2,
                                       fill=color, outline="")
            else:
                c.create_text(W//2, H//2, text="Waveform appears during recording",
                               fill=PALETTE["muted"], font=("Segoe UI", 11))
        except Exception:
            pass
        self._waveform_job = self.after(50, self._draw_waveform)

    # ──────────────────────────────────────────────────────────────────────
    # Timer
    # ──────────────────────────────────────────────────────────────────────
    def _start_timer(self):
        self._timer_seconds = 0
        self._tick_timer()

    def _tick_timer(self):
        if self._is_recording:
            self._timer_seconds += 1
            m = self._timer_seconds // 60
            s = self._timer_seconds %  60
            self._timer_label.configure(text=f"{m:02d}:{s:02d}")
            self._timer_job = self.after(1000, self._tick_timer)

    def _stop_timer(self):
        if self._timer_job:
            self.after_cancel(self._timer_job)

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────
    def _set_status(self, text: str, color: str):
        self._status_dot.configure(text_color=color)
        self._status_label.configure(text=text, text_color=color)

    def _show_progress(self, show: bool):
        if show:
            self._progress_frame.pack(fill="x", after=self._wave_frame)
        else:
            self._progress_frame.pack_forget()
            self._progress_bar.set(0)
            self._progress_label.configure(text="")

    def _clear_results(self):
        self._session_data.clear()
        self._tree.delete(*self._tree.get_children())
        for w in self._spk_scroll.winfo_children(): w.destroy()
        self._spk_widgets.clear()
        self._tl_canvas.delete("all")
        self.ai_ollama_model.configure(text="—")
        self.ai_alt_model.configure(text="—")
        self.ai_best_source.configure(text="—")
        self.ai_ollama_text.configure(text="—")
        self.ai_alt_text.configure(text="—")
        self.ai_best_text.configure(text="—")
        self._reason_lbl.configure(text="—")

    # ──────────────────────────────────────────────────────────────────────
    # Excel Export
    # ──────────────────────────────────────────────────────────────────────
    def _export_excel(self):
        if not self._session_data:
            mb.showinfo("Nothing to export", "Process a recording first.")
            return
        path = fd.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel Workbook", "*.xlsx")],
            initialfile=f"SpeakSense_{datetime.datetime.now():%Y%m%d_%H%M%S}.xlsx",
            title="Save Excel File")
        if not path: return
        try:
            export_excel(self._session_data, path)
            mb.showinfo("Exported!", f"Saved to:\n{path}")
        except Exception as e:
            mb.showerror("Export failed", str(e))


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = SpeakSenseApp()
    app.mainloop()      