# VoiceLLM

Fully local, low-latency voice assistant running entirely on Apple Silicon (M3 Pro, 18 GB). No cloud, no API keys, no internet required.

**Round-trip latency: ~4.5–6s** (warm models) — STT 0.5s · LM 3s · TTS 1s

---

## Architecture

```mermaid
flowchart TD
    MIC["🎙 Microphone\npush-to-talk"]
    STT["Parakeet TDT 0.6B-v3\nNeMo · MPS\nRTF 0.10–0.20x"]
    LM["Qwen3.5:4b\nOllama · /api/chat\nconversation context"]
    TTS["Kokoro-82M\nff_siwis · CPU\nRTF 0.19–0.21x"]
    SPK["🔊 Speaker\nafplay"]

    MIC -->|"16kHz mono WAV"| STT
    STT -->|"raw transcript\n(FR/EN, may have errors)"| LM
    LM -->|"clean response\n2 phrases max"| TTS
    TTS -->|"24kHz WAV"| SPK

    style MIC fill:#1e1e2e,color:#cdd6f4,stroke:#585b70
    style STT fill:#313244,color:#cdd6f4,stroke:#585b70
    style LM  fill:#313244,color:#cdd6f4,stroke:#585b70
    style TTS fill:#313244,color:#cdd6f4,stroke:#585b70
    style SPK fill:#1e1e2e,color:#cdd6f4,stroke:#585b70
```

---

## Models

| Component | Model | Backend | VRAM | RTF |
|-----------|-------|---------|------|-----|
| STT | [Parakeet TDT 0.6B-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | NeMo / MPS | ~0.9 GB | 0.10–0.20x |
| LM | [Qwen3.5:4b](https://ollama.com/library/qwen3.5) | Ollama | ~3.4 GB | — |
| TTS | [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | PyTorch / CPU | ~0.3 GB | 0.19–0.21x |
| **Total** | | | **~4.6 GB** | |

---

## Features

- 🇫🇷🇬🇧 **Bilingual** — French, English, and FR/EN code-switching handled natively
- 🧠 **Conversation context** — 5-turn sliding window, auto-reset after 5 minutes
- ⚡ **Fast** — all models loaded once at startup, kept in memory
- 🔒 **Fully offline** — zero network calls during inference
- 📋 **Timestamped logs** — every run logged to `logs/`

---

## Requirements

- macOS (Apple Silicon recommended — M1/M2/M3)
- Python 3.11+
- [Ollama](https://ollama.com) running locally with `qwen3.5:4b` pulled

---

## Installation

```bash
# Clone
git clone <repo-url>
cd VoiceLLM

# Dependencies
pip install kokoro soundfile sounddevice httpx
pip install nemo_toolkit[asr]

# Pull LM
ollama pull qwen3.5:4b
```

---

## Usage

### Full pipeline (recommended)

```bash
python pipeline.py
```

Press **Enter** to start recording, **Enter** again to stop. The pipeline transcribes, generates a response, and plays it back.

### Individual components

```bash
# TTS only
python tts.py "Bonjour, comment ça va ?"
python tts.py --test

# LM only
python lm.py "c'est quoi un trou noir"
python lm.py --test
python lm.py --wire "quelle heure il est"   # LM → TTS → audio

# STT only
python stt.py                   # push-to-talk
python stt.py --file audio.wav  # transcribe a file
```

---

## File Structure

```
VoiceLLM/
├── pipeline.py      # full loop — mic → STT → LM → TTS → speaker
├── stt.py           # Parakeet STT wrapper
├── lm.py            # Ollama LM wrapper (stateless + chat history)
├── tts.py           # Kokoro TTS wrapper
├── log_utils.py     # shared timestamped logging
├── bench_mlx.py     # TTS backend benchmark (llama.cpp vs mlx-audio)
└── CLAUDE.md        # architecture decisions and gotchas
```

---

## Latency Profile (M3 Pro, warm models)

| Stage | Typical | Notes |
|-------|---------|-------|
| STT (Parakeet) | 0.5–0.8s | for a 4–10s utterance |
| LM (Qwen3.5:4b) | 1.5–3.5s | depends on response length |
| TTS (Kokoro) | 0.9–2.0s | RTF ~0.2x |
| **Round-trip** | **~4–6s** | warm, short response |

Cold start (first run): +3–5s for Ollama model load.

---

## Roadmap

- [x] Phase 1 — TTS standalone (Kokoro-82M)
- [x] Phase 2 — LM integration (Qwen3.5:4b via Ollama)
- [x] Phase 3 — STT integration (Parakeet TDT 0.6B-v3)
- [ ] Phase 4 — Web grounding (DuckDuckGo + Wikipedia, no API key)

---

## License

MIT
