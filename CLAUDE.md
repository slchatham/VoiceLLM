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

## 6. Phase 4 — Web grounding ✅

**Goal**: give the LM access to real-time factual data for queries it cannot answer locally (current events, people, dates, prices, etc.).

**Result**: Ollama native tool calling — `web_search` (DuckDuckGo lite→html fallback), `wikipedia_lookup`, `get_stock_price` (yfinance). LM triggers tools autonomously. Domain blocklist for hoax/satire sites. Tool queries always in English for better coverage. DDG ratelimit handled via lite→html fallback. `num_predict=1200` cap on think mode to prevent runaways.

### Go / No-Go — PASSED
- Tool calls triggered autonomously, correct results ✓
- DDG fallback working, +1.5s overhead on ratelimit ✓
- Stock prices: real OHLCV data, parallel tool calls (TSLA + NVDA in one pass) ✓
- Round-trip with tools: ~8–12s warm ✓

---

## 7. Phase 5 — RAG Local (ready to implement)

**Goal**: allow the LM to query structured local files (spreadsheets, CSV) via a `query_collection` tool.

**Branch**: `feature/rag-collection`

**Architecture**:
- `documents/` — input folder, scanned at startup
- `index_documents.py` — standalone indexer, MD5 hash to skip unchanged files, each file becomes a SQLite table named after the filename
- `tools/rag.py` — the tool: Qwen generates SQL from question + injected schema, executes on `collection.db`, returns formatted result
- Schema injection format: table list + column names + types + 3 example values per table — enough context for a 4b to write correct SQL
- `lm.py`: add `query_collection` to `tools.DEFINITIONS`, same pattern as existing tools

**Formats supported**: `.xlsx`, `.xls`, `.csv`

**Validated test case**: board game collection (380 entries) — "quels jeux je n'ai pas joués ?", "combien dépensé chez Philibert ?", "jeux de Reiner Knizia que j'ai ?"

**Key constraint**: inject the full list of available tables (not just the queried one) so the LM can choose the right table or JOIN across files.

**New dependency**: `openpyxl` (xlsx read), `pandas` (CSV/XLS → SQLite ingestion)

---

## 8. Phase 6 — Dynamic Persona Detection (architecture decided)

**Goal**: adapt the system prompt in real time based on detected conversation domain — cybersecurity architect, financial advisor, business coach, etc.

**Key architectural constraint**: changing the system prompt invalidates Ollama's KV cache prefix, costing ~3–5s re-warm on the first turn after a switch. To preserve the cache, **persona is injected as a hint prefix in the user message**, not as a system prompt replacement:
```python
augmented = f"[mode: {persona}]\n{user_message}"
```

**Detection**: embeddings (nomic-embed-text via Ollama, ~0.3 GB) + cosine similarity to pre-computed persona centroid vectors. Cost: <100ms, zero tokens consumed. Centroid vectors pre-computed at startup from descriptive text for each tag.

**Switch policy**: detect every turn, but only switch if the new persona is stable for 2 consecutive turns — avoids oscillation on ambiguous questions.

**Persona mode**: mono-persona dominant (no stacking — system prompt inflation not worth it for a 4b).

**Persona tags**: `cybersecurity_architect`, `financial_advisor`, `business_coach`, `lifestyle_casual`, `board_game_expert`, `technical_developer`, `health_wellness`, `general_assistant`

**CLI**:
```
--persona auto    # dynamic detection (default)
--persona fixed   # static system prompt (current behavior)
--persona <tag>   # force a specific persona
```

**New dependency**: `nomic-embed-text` pulled in Ollama (or `mxbai-embed-large`), no pip package needed

---

## 9. Known Quirks & Gotchas

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
- **Ctrl+C exit**: use `os._exit(0)` instead of `break` in the main loop — NeMo/MPS tensors crash during Python destructor teardown on macOS. `os._exit(0)` skips destructors entirely.
- **Tool calling**: `lm.py ask()` passes `tools.DEFINITIONS` on every `/api/chat` call. If `message.tool_calls` is present in the stream, tools are executed and a second streaming call is made. Tool internals (calls + results) are NOT added to the sliding history — only the final user/assistant exchange.
- **Warmup call**: `pipeline.py` makes a `ask(".", ..., tools=[])` call at startup (after model load) to prime Ollama's KV cache for the system prompt. `tools=[]` prevents the model from triggering a spurious tool call on the warmup message. Cost: ~5s at startup, saves ~7–9s on the first real turn.
- **KV cache and persona switching**: changing the system prompt between turns invalidates Ollama's prefix cache. Any dynamic persona feature must inject persona context into the user message, not the system prompt.
- **web_search is unreliable for real-time weather** — DDG returns news snippets, not live data. Abstain or add a dedicated weather API tool.
- **European stock tickers** need a market suffix: `AXA.PA`, `MC.PA` (Euronext Paris), `SAP.DE` (Xetra). The LM knows the major ones but may miss smaller caps.
- **Parakeet hallucination on noisy audio** — on poor mic input, Parakeet may generate fluent text in an unexpected language (observed: Russian). The LM will respond in that language (correct per language rules). No fix in current code — user must re-speak clearly.

---

## 10. Stack & Dependencies

```
kokoro                # TTS — Kokoro-82M, French voice ff_siwis
soundfile             # WAV I/O for Kokoro output
ollama (httpx)        # LM serving (Qwen3.5:4b) — direct HTTP via httpx
nemo_toolkit          # STT (Parakeet TDT 0.6B-v3)
sounddevice           # mic capture
numpy                 # audio buffers
duckduckgo-search     # web_search tool (no API key)
wikipedia             # wikipedia_lookup tool (no API key)
yfinance              # get_stock_price tool (no API key)
```

```bash
# Phase 1
pip install kokoro soundfile sounddevice

# Phase 2 (Ollama must already be running with qwen3.5:4b pulled)
pip install httpx

# Phase 3
pip install nemo_toolkit[asr]

# Phase 4
pip install duckduckgo-search wikipedia yfinance

# Phase 5 (planned)
pip install openpyxl pandas

# Phase 6 (planned)
ollama pull nomic-embed-text
```

---

## 11. File Structure

```
VoiceLLM/
├── CLAUDE.md              # this file
├── pipeline.py            # full loop — mic → STT → LM → TTS → speaker
├── stt.py                 # Phase 3 — Parakeet STT wrapper
├── lm.py                  # Phase 2 — Ollama wrapper (stateless + chat history + tool calling)
├── tts.py                 # Phase 1 — Kokoro TTS wrapper
├── tools.py               # Phase 4 — DuckDuckGo + Wikipedia + yfinance tools
├── log_utils.py           # shared timestamped logging
├── bench_mlx.py           # TTS backend benchmark (kept for reference)
├── documents/             # Phase 5 — input files for RAG (xlsx, csv) [planned]
├── index_documents.py     # Phase 5 — file indexer → SQLite [planned]
├── tools/
│   └── rag.py             # Phase 5 — query_collection tool [planned]
├── speakers/              # OuteTTS speaker profiles (kept for reference)
│   └── fr_voice.json
└── output/                # generated WAV files (gitignored)
```

---

## 12. What We Are NOT Doing (for now)

- **No UI** — CLI only until all phases validate
- **No streaming TTS** — full LM response collected before TTS starts (simplicity first)
- **No wake word** — push-to-talk or manual trigger only
- **No Voxtral / vllm** — requires NVIDIA GPU, incompatible with M3 Pro MPS
- **No packaging** — this is a prototype, not a product (yet)
