# VoiceLLM — CLAUDE.md

Local voice pipeline: Speech-to-Text → Language Model → Text-to-Speech, fully offline on Apple Silicon.

---

## 1. Objective & Architecture

Build a fully local, low-latency voice loop on macOS (M3 Pro):

```
Microphone
    │
    ▼
[Parakeet TDT 0.6B-v3]  ← NeMo / MPS
    │  raw transcript (may contain errors, code-switching FR/EN)
    ▼
[Qwen3.5:4b via Ollama]  ← small LM, correction + response (with conversation context)
    │  clean text response
    ▼
[Kokoro-82M]  ← PyTorch / CPU, ff_siwis French voice
    │  synthesized speech
    ▼
Speaker output
```

Each block is developed and validated in isolation before integration.

---

## 2. Constraints

- **Hardware**: MacBook Pro M3 Pro, 18 GB unified memory
- **All models must fit simultaneously**: target < 8 GB total, hard limit 18 GB
- **Fully local**: no network calls, no cloud APIs, no external services (except Phase 4 web grounding)
- **Python 3.10+**, minimal dependencies per phase
- **No UI in early phases** — CLI only until all three blocks are validated

### Memory budget

| Component         | Approx VRAM |
|-------------------|-------------|
| Parakeet TDT 0.6B | ~0.9 GB     |
| Qwen3.5:4b        | ~3.4 GB     |
| Kokoro-82M        | ~0.3 GB     |
| **Total**         | **~4.6 GB** |

---

## 3. Phase 1 — TTS standalone ✅

**Goal**: validate TTS quality, latency, and French language output on M3 Pro.

**Result**: OuteTTS rejected (stochastic quality, RTF 5x). Replaced by **Kokoro-82M** — deterministic, RTF 0.19-0.21x, French voice `ff_siwis` natively supported.

### Go / No-Go — PASSED
- French speech intelligible and natural
- Latency < 2s on M3 Pro ✓ (RTF ~0.2x)
- No crashes or memory pressure ✓

---

## 4. Phase 2 — Ollama LM in the middle ✅

**Goal**: insert Qwen3.5:4b between STT output and TTS. Handles correction, response generation, and conversation context.

**Key decisions**:
- `/api/chat` endpoint for conversation history (sliding window: 5 turns, 300s timeout reset)
- `think: False` + `_ThinkFilter` to strip reasoning tokens
- 2 phrases max in system prompt — LM keeps responses concise without rigid word count constraints

### Go / No-Go — PASSED
- No hallucination on correction tasks ✓
- Round-trip LM + TTS ~5s warm ✓
- FR/EN code-switching preserved ✓

---

## 5. Phase 3 — Parakeet STT input ✅

**Goal**: replace hardcoded text input with live microphone transcription via Parakeet TDT.

**Result**: push-to-talk loop in `pipeline.py`. All three models loaded at startup, kept in memory for the session.

### Go / No-Go — PASSED
- Full round-trip 4.5–6s warm ✓
- No memory pressure ✓
- FR/EN code-switching handled correctly ✓

---

## 6. Phase 4 — Web grounding (planned)

**Goal**: give the LM access to real-time factual data for queries it cannot answer locally (current events, people, dates, prices, etc.).

**Motivation**: Qwen3.5:4b hallucinate on recent facts without grounding (observed: Macron/Brune Payer, timestamps, current events).

**Approach**:
- Tools: `duckduckgo-search` + `wikipedia-api` — free, no API key required
- Trigger: LM decides itself if a search is needed via a flag in the system prompt — no function calling framework
- Injection: search results injected into the `/api/chat` messages array as a `system` message before the final response turn
- Estimated additional latency: +1–2s per round-trip when search is triggered

**Limitations to document**:
- Search adds latency — only trigger when necessary
- DuckDuckGo results may be stale or irrelevant; inject with a disclaimer in the prompt
- Wikipedia snippets are truncated — do not exceed context budget

### Tasks
- [ ] Install `duckduckgo-search` and `wikipedia-api`
- [ ] Define trigger heuristic in system prompt (e.g. "if you need current data, output `[SEARCH: query]` on the first line")
- [ ] Parse LM first-pass output for `[SEARCH: ...]` flag
- [ ] Fetch and truncate results
- [ ] Re-inject as context and get final response
- [ ] Measure latency overhead

---

## 7. Known Quirks & Gotchas

### Kokoro
- **Replaces OuteTTS as the primary TTS** — OuteTTS was stochastic (bad output ~50% of the time on M3 Pro). Kokoro is deterministic, RTF 0.19–0.21x, empirically validated.
- **French voice**: `ff_siwis`, `lang_code="f"`. Handles FR/EN code-switching well with a light French accent on English words.
- **No sampling** — output is deterministic, no temperature, no clipping issues.

### Parakeet TDT
- **Push-to-talk**: short utterances (3–15s) work well. The 60s chunk size from Dictation Corrector is for continuous dictation, not needed here.
- **NeMo logger spam** on every `transcribe()` call — suppressed via `_silence_nemo_loggers()`, must be called after model load AND before each transcription call.
- Returns naive datetime objects from NeMo — always localize to UTC when timestamping.

### Ollama / Qwen3.5:4b
- Model `qwen3.5:4b` is available locally — no pull needed.
- Use `/api/chat` (not `/api/generate`) for conversation context — different response field: `data["message"]["content"]` vs `data["response"]`.
- `think: False` in the request body disables reasoning tokens. `_ThinkFilter` is a safety net in case any slip through.
- Cold load: ~3s on first call. Warm: ~0.15s. Keep Ollama running between sessions.

### General
- Load all three models at startup, not on demand — cold load latency is unacceptable mid-conversation.
- Conversation history: 5-turn sliding window, resets after 300s of silence.
- Test each phase independently with `--test` flag before full pipeline run.

---

## 8. Stack & Dependencies

```
kokoro           # TTS — Kokoro-82M, French voice ff_siwis
soundfile        # WAV I/O for Kokoro output
ollama (httpx)   # LM serving (Qwen3.5:4b) — direct HTTP via httpx
nemo_toolkit     # STT (Parakeet TDT 0.6B-v3)
sounddevice      # mic capture
numpy            # audio buffers
```

```bash
# Phase 1
pip install kokoro soundfile sounddevice

# Phase 2 (Ollama must already be running with qwen3.5:4b pulled)
pip install httpx

# Phase 3
pip install nemo_toolkit[asr]

# Phase 4 (planned)
pip install duckduckgo-search wikipedia-api
```

---

## 9. File Structure

```
VoiceLLM/
├── CLAUDE.md              # this file
├── tts.py                 # Phase 1 — Kokoro TTS wrapper
├── lm.py                  # Phase 2 — Ollama wrapper (stateless + chat history)
├── stt.py                 # Phase 3 — Parakeet wrapper
├── pipeline.py            # full loop orchestration (Phases 1–3)
├── log_utils.py           # shared logging setup
├── speakers/              # OuteTTS speaker profiles (kept for reference)
│   └── fr_voice.json
├── voice_reference/       # source audio for speaker cloning
└── output/                # generated WAV files (gitignored)
```

---

## 10. What We Are NOT Doing (for now)

- **No UI** — CLI only until all phases validate
- **No streaming TTS** — full LM response collected before TTS starts (simplicity first)
- **No wake word** — push-to-talk or manual trigger only
- **No Voxtral / vllm** — requires NVIDIA GPU, incompatible with M3 Pro MPS
- **No packaging** — this is a prototype, not a product (yet)
