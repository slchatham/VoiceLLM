# VoiceLLM

Fully local, low-latency voice assistant running entirely on Apple Silicon (M3 Pro, 18 GB). No cloud, no API keys — web grounding optional via DuckDuckGo + Wikipedia.

**Round-trip latency: ~4–6s** (warm, qwen3.5:4b, no tools) · **~8–10s** (with tool call)

---

## Architecture

```mermaid
flowchart TD
    MIC["🎙 Microphone\npush-to-talk"]
    STT["Parakeet TDT 0.6B-v3\nNeMo · MPS\nRTF 0.10–0.20x"]
    LM["Qwen3.5:4b (default)\nOllama · /api/chat\nconversation context + tools"]
    TOOLS["DuckDuckGo · Wikipedia · yfinance\nweb grounding (optional)"]
    TTS["Kokoro-82M\nff_siwis (FR) · af_heart (EN)\nRTF 0.19–0.22x"]
    SPK["🔊 Speaker\nafplay"]

    MIC -->|"16kHz mono WAV"| STT
    STT -->|"raw transcript\n(FR/EN, may have errors)"| LM
    LM <-->|"tool calls\n(when needed)"| TOOLS
    LM -->|"clean response\n2 phrases max"| TTS
    TTS -->|"24kHz WAV"| SPK

    style MIC   fill:#1e1e2e,color:#cdd6f4,stroke:#585b70
    style STT   fill:#313244,color:#cdd6f4,stroke:#585b70
    style LM    fill:#313244,color:#cdd6f4,stroke:#585b70
    style TOOLS fill:#45475a,color:#cdd6f4,stroke:#585b70
    style TTS   fill:#313244,color:#cdd6f4,stroke:#585b70
    style SPK   fill:#1e1e2e,color:#cdd6f4,stroke:#585b70
```

---

## Models

| Component | Model | Backend | VRAM | RTF |
|-----------|-------|---------|------|-----|
| STT | [Parakeet TDT 0.6B-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | NeMo / MPS | ~0.9 GB | 0.10–0.20x |
| LM | [Qwen3.5:4b](https://ollama.com/library/qwen3.5) *(default)* | Ollama | ~3.4 GB | — |
| LM | [Qwen3.5:9b](https://ollama.com/library/qwen3.5) *(optional, slow)* | Ollama | ~6.0 GB | — |
| LM | [Qwen3.5:2b](https://ollama.com/library/qwen3.5) *(experimental)* | Ollama | ~1.8 GB | — |
| TTS | [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | PyTorch / CPU | ~0.3 GB | 0.19–0.22x |
| **Total (4b)** | | | **~4.6 GB** | |
| **Total (9b)** | | | **~7.2 GB** | |

---

## Features

- 🇫🇷🇬🇧 **Bilingual** — French, English, and FR/EN code-switching handled natively
- 🗣 **Dynamic TTS voice routing** — `ff_siwis` for French responses, `af_heart` for English, detected automatically from LM output using a shared Kokoro model (weights loaded once)
- 🧠 **Conversation context** — 5-turn sliding window, auto-reset after 5 minutes
- ⚡ **Fast** — all models loaded once at startup, kept in memory
- 🌐 **Web grounding** — LM triggers DuckDuckGo / Wikipedia / yfinance tool calls autonomously when needed; +3–4s overhead, zero manual trigger
- 📈 **Stock prices** — `get_stock_price(ticker, period)` via yfinance; real OHLCV data for 1d / 7d / 1mo / 3mo; LM resolves company → ticker (NVDA, MC.PA, etc.)
- 🔒 **Fully offline by default** — tool calls are the only optional network path
- 📋 **Timestamped logs** — every run logged to `logs/`
- 🎛 **Selectable LM** — `--model qwen3.5:4b` (default) · `9b` (slower, higher quality) · `2b` (experimental, fast but hallucinates with tools)

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
pip install duckduckgo-search wikipedia yfinance   # Phase 4 — web grounding + stock prices

# Pull LM (4b — default)
ollama pull qwen3.5:4b

# Pull LM (optional)
ollama pull qwen3.5:9b   # higher quality, slower
ollama pull qwen3.5:2b   # experimental
```

---

## Usage

### Full pipeline (recommended)

```bash
python pipeline.py                        # qwen3.5:4b (default)
python pipeline.py --model qwen3.5:9b    # higher quality, ~10–25s round-trip
python pipeline.py --model qwen3.5:2b    # experimental — fast but unreliable with tools
python pipeline.py --no-play             # transcribe + LM only, no audio output
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
python lm.py --model qwen3.5:9b --test
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
├── lm.py            # Ollama LM wrapper (stateless + chat history + tool calling)
├── tts.py           # Kokoro TTS wrapper
├── tools.py         # DuckDuckGo + Wikipedia tool definitions and execution
├── log_utils.py     # shared timestamped logging
├── bench_mlx.py     # TTS backend benchmark (llama.cpp vs mlx-audio)
└── CLAUDE.md        # architecture decisions and gotchas
```

---

## Latency Profile (M3 Pro, warm models)

### qwen3.5:4b — recommended for real-time

| Stage | Typical | Notes |
|-------|---------|-------|
| STT (Parakeet) | 0.5–0.8s | for a 4–10s utterance |
| LM (Qwen3.5:4b) | 1.5–3.5s | depends on response length |
| TTS (Kokoro) | 0.9–2.0s | RTF ~0.2x |
| **Round-trip** | **~4–6s** | warm, short response |

### qwen3.5:9b — better quality, slower

| Stage | Typical | Notes |
|-------|---------|-------|
| STT (Parakeet) | 1–4s | degraded — memory pressure |
| LM (Qwen3.5:9b) | 5–15s | warm; first call ~13s |
| TTS (Kokoro) | 2–5s | RTF ~0.2–0.4x |
| **Round-trip** | **~10–25s** | not recommended for real-time |

Cold start (first run): +3–5s for Ollama model load.

---

## Decision Log

| Decision | Rejected | Chosen | Reason |
|----------|----------|--------|--------|
| TTS engine | OuteTTS (llama.cpp Q8_0) | **Kokoro-82M** | OuteTTS produced distorted audio ~50% of the time on M3 Pro — stochastic sampling instability. Kokoro is deterministic, RTF 0.19–0.21x, clean French output. |
| OuteTTS backend | mlx-audio BF16 (RTF 3.7x) | llama.cpp Q8_0 (RTF 3.1x) | Benchmarked in `bench_mlx.py` — llama.cpp faster, but both superseded by Kokoro. |
| LM size | Qwen3.5:2b | **Qwen3.5:4b** | 2b hallucinated facts and ignored language rules. 4b stable with no observable latency regression. |
| LM API | `/api/generate` | **`/api/chat`** | `/api/generate` is stateless. `/api/chat` supports conversation history natively. |
| TTS lang routing | single pipeline (FR only) | **dual pipeline, shared KModel** | `KPipeline.lang_code` is fixed at init. Sharing one `KModel` avoids loading weights twice. FR→`ff_siwis`, EN→`af_heart`, detected via function word heuristic. |
| Default LM size | 9b | **4b** | 9b causes STT degradation (RTF 1.0x vs 0.1x) and 10–25s round-trips due to memory pressure. 4b hits the sweet spot at ~5s. |
| Web grounding approach | prompt-flag `[SEARCH: ...]` | **Ollama tool calling** | Native function calling is cleaner — model decides autonomously, no regex parsing, no manual trigger heuristic. |
| 2b + tools viability | qwen3.5:2b with tools | **4b (default)** | 2b calls tools correctly but fails to synthesize results — hallucinates on top of correct search results. Fast (~1.4s LM) but unreliable. Left available as `--model qwen3.5:2b`. |
| Stock price tool | DuckDuckGo snippets | **yfinance** | DDG returns stale financial news snippets with inconsistent numbers. yfinance gives real OHLCV data with accurate % change over any period. |
| Ctrl+C exit crash | `break` → Python cleanup | **`os._exit(0)`** | NeMo/MPS tensors crash during Python destructor teardown on macOS. `os._exit(0)` skips destructors entirely — clean exit, no bus error. |

---

## Roadmap

- [x] Phase 1 — TTS standalone (Kokoro-82M)
- [x] Phase 2 — LM integration (Qwen3.5:4b via Ollama)
- [x] Phase 3 — STT integration (Parakeet TDT 0.6B-v3)
- [x] Phase 3.5 — Dynamic FR/EN TTS routing (shared KModel)
- [x] Phase 3.6 — Selectable LM (`--model qwen3.5:2b|4b|9b`)
- [x] Phase 4 — Web grounding (DuckDuckGo + Wikipedia + yfinance via Ollama tool calling)
- [ ] Phase 5 — Dedicated weather tool (OpenWeatherMap or equivalent — DDG unreliable for real-time weather)

---

## License

MIT
