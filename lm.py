"""
Phase 2 — Ollama LM wrapper (qwen3.5:4b / 9b)
raw_text → ollama HTTP API → clean_text

Usage:
    python lm.py "j'ai demandé kelle heure il était"
    python lm.py --test
    python lm.py --wire "quelle heure il est"   # LM → TTS → plays audio
    python lm.py --model qwen3.5:9b --test
"""

import argparse
import json
import os
import time

import httpx

import log_utils
import tools as _tools

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_URL      = "http://localhost:11434/api/generate"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL             = "qwen3.5:4b"
AVAILABLE_MODELS  = ["qwen3.5:2b", "qwen3.5:4b", "qwen3.5:9b"]
THINK_MAX_TOKENS  = 1200  # cap total tokens (think + response) in think=ON mode

_SYSTEM_PROMPT_TEMPLATE = """\
You are a fully local voice assistant running on a Mac.
You have access to web search (DuckDuckGo) and Wikipedia tools for questions requiring recent or factual data.
Today's date is {date}.

Absolute rules:
- ALWAYS reply in the same language as the user's message. If the message is in French, reply in French. If in English, reply in English. Never translate.
- If the message mixes French and English, preserve that mix in your reply.
- Reply in 2 sentences maximum. Be natural and direct.
- Declarative sentences only. No decorative punctuation.
- Use tools for news, prices, recent facts, biographies, or any data you cannot confirm with certainty.
- When calling any tool (web_search, wikipedia_lookup), ALWAYS write the query in English — results are better. The final response stays in the user's language.
- When you have tool results, present the data directly and concisely. NEVER say "Je vérifie", "I am checking", or describe what you are doing — the user wants the answer, not a commentary on your process.
- If you genuinely don't know something and no tool can help, say so in one sentence and stop.
- You do not have access to the exact time or the user's location — say so honestly if asked.
- If the text contains obvious speech-to-text transcription errors, silently correct them without mentioning it.\
"""


def _system_prompt() -> str:
    import datetime
    date = datetime.date.today().strftime("%d %B %Y")
    return _SYSTEM_PROMPT_TEMPLATE.format(date=date)

TEST_INPUTS = [
    "kelle heure il est",                          # FR with STT error
    "what's the weather like today",               # EN
    "j'ai besoin d'aide pour my presentation",     # FR/EN code-switch
    "dis moi un truc intéressant sur les pieuvres",
]

# ---------------------------------------------------------------------------
# <think>…</think> filter  (Qwen3.5 reasoning tokens, safety net)
# ---------------------------------------------------------------------------

class _ThinkFilter:
    _OPEN  = "<think>"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._buf = ""
        self._in  = False

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        out: list[str] = []
        while self._buf:
            if self._in:
                p = self._buf.find(self._CLOSE)
                if p >= 0:
                    self._buf = self._buf[p + len(self._CLOSE):]
                    self._in  = False
                else:
                    self._buf = ""
                    break
            else:
                p = self._buf.find(self._OPEN)
                if p >= 0:
                    out.append(self._buf[:p])
                    self._buf = self._buf[p + len(self._OPEN):]
                    self._in  = True
                else:
                    for i in range(1, len(self._OPEN)):
                        if self._buf.endswith(self._OPEN[:i]):
                            out.append(self._buf[:-i])
                            self._buf = self._buf[-i:]
                            break
                    else:
                        out.append(self._buf)
                        self._buf = ""
                    break
        return "".join(out)

    def flush(self) -> str:
        val, self._buf = ("" if self._in else self._buf), ""
        return val


# ---------------------------------------------------------------------------
# LM call
# ---------------------------------------------------------------------------

def ask(raw_text: str, log, history: list[dict] | None = None, model: str = MODEL, think: bool = False) -> tuple[str, float]:
    """Send raw_text to the LM.

    history: list of {"role": "user"|"assistant", "content": "..."} messages.
             If provided, uses /api/chat to maintain conversation context.
             The new exchange (user + assistant) is appended to history in-place.
             If None, uses /api/generate (stateless).
    model:   Ollama model name (default: MODULE-level MODEL constant).

    Returns (response, elapsed_seconds).
    """
    log.info(f"LM input : {raw_text!r}  (context: {len(history)} msgs)" if history else f"LM input : {raw_text!r}")
    t0  = time.perf_counter()
    flt = _ThinkFilter()
    parts: list[str] = []
    first_token_logged = False

    try:
        with httpx.Client(timeout=60.0) as cli:
            if history is not None:
                # Chat mode — full conversation context + tool calling
                messages = (
                    [{"role": "system", "content": _system_prompt()}]
                    + history
                    + [{"role": "user", "content": raw_text}]
                )
                log.debug(f"POST {OLLAMA_CHAT_URL} model={model} messages={len(messages)}")

                # ── First pass: stream, detect tool calls ────────────────────
                tool_calls: list[dict] = []
                with cli.stream("POST", OLLAMA_CHAT_URL, json={
                    "model":    model,
                    "messages": messages,
                    "tools":    _tools.DEFINITIONS,
                    "stream":   True,
                    "think":    think,
                    **({"options": {"num_predict": THINK_MAX_TOKENS}} if think else {}),
                }) as resp:
                    log.debug(f"HTTP {resp.status_code}")
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        data = json.loads(line)
                        msg  = data.get("message", {})
                        tcs  = msg.get("tool_calls")
                        if tcs:
                            tool_calls.extend(tcs)
                        chunk = flt.feed(msg.get("content", "") or "")
                        if chunk:
                            if not first_token_logged:
                                log.info(f"first token in {time.perf_counter() - t0:.2f}s")
                                first_token_logged = True
                            parts.append(chunk)
                        if data.get("done"):
                            eval_count    = data.get("eval_count", 0)
                            eval_duration = data.get("eval_duration", 0) / 1e9
                            load_duration = data.get("load_duration", 0) / 1e9
                            log.info(f"done — load: {load_duration:.2f}s | eval: {eval_duration:.2f}s | {eval_count} tokens")
                            break
                    tail = flt.flush()
                    if tail:
                        parts.append(tail)

                # ── Second pass: execute tools, get final response ───────────
                if tool_calls:
                    assistant_msg = {
                        "role":       "assistant",
                        "content":    "".join(parts).strip(),
                        "tool_calls": tool_calls,
                    }
                    tool_messages = [assistant_msg]
                    for tc in tool_calls:
                        fn     = tc.get("function", {})
                        result = _tools.execute(fn.get("name", ""), fn.get("arguments", {}), log)
                        tool_messages.append({"role": "tool", "content": result})

                    parts = []
                    flt   = _ThinkFilter()
                    first_token_logged = False

                    log.debug(f"POST {OLLAMA_CHAT_URL} model={model} messages={len(messages)+len(tool_messages)} [after tools]")
                    with cli.stream("POST", OLLAMA_CHAT_URL, json={
                        "model":    model,
                        "messages": messages + tool_messages,
                        "stream":   True,
                        "think":    think,
                        **({"options": {"num_predict": THINK_MAX_TOKENS}} if think else {}),
                    }) as resp2:
                        resp2.raise_for_status()
                        for line in resp2.iter_lines():
                            if not line:
                                continue
                            data  = json.loads(line)
                            chunk = flt.feed(data.get("message", {}).get("content", "") or "")
                            if chunk:
                                if not first_token_logged:
                                    log.info(f"first token (post-tool) in {time.perf_counter() - t0:.2f}s")
                                    first_token_logged = True
                                parts.append(chunk)
                            if data.get("done"):
                                eval_count    = data.get("eval_count", 0)
                                eval_duration = data.get("eval_duration", 0) / 1e9
                                log.info(f"done (post-tool) — eval: {eval_duration:.2f}s | {eval_count} tokens")
                                break
                        tail = flt.flush()
                        if tail:
                            parts.append(tail)
            else:
                # Stateless mode — /api/generate (used by --test and --wire)
                log.debug(f"POST {OLLAMA_URL} model={model}")
                with cli.stream("POST", OLLAMA_URL, json={
                    "model":  model,
                    "prompt": raw_text,
                    "system": _system_prompt(),
                    "stream": True,
                    "think":  False,
                }) as resp:
                    log.debug(f"HTTP {resp.status_code}")
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        data  = json.loads(line)
                        chunk = flt.feed(data.get("response", ""))
                        if chunk:
                            if not first_token_logged:
                                log.info(f"first token in {time.perf_counter() - t0:.2f}s")
                                first_token_logged = True
                            parts.append(chunk)
                        if data.get("done"):
                            eval_count    = data.get("eval_count", 0)
                            eval_duration = data.get("eval_duration", 0) / 1e9
                            load_duration = data.get("load_duration", 0) / 1e9
                            log.info(f"done — load: {load_duration:.2f}s | eval: {eval_duration:.2f}s | {eval_count} tokens")
                            break
                    tail = flt.flush()
                    if tail:
                        parts.append(tail)

    except httpx.ConnectError:
        log.error(f"cannot reach Ollama — is 'ollama serve' running?")
        raise
    except Exception as e:
        log.error(f"{type(e).__name__}: {e}")
        raise

    elapsed = time.perf_counter() - t0
    text    = "".join(parts).strip()
    log.info(f"LM output: {text!r}  ({elapsed:.2f}s total)")

    if history is not None:
        history.append({"role": "user",      "content": raw_text})
        history.append({"role": "assistant", "content": text})

    return text, elapsed


# ---------------------------------------------------------------------------
# Wire: LM → TTS → audio
# ---------------------------------------------------------------------------

def wire(raw_text: str):
    """LM → TTS → play. Measures each stage."""
    import subprocess
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    log = log_utils.setup("wire")
    log.info(f"=== wire start ===  input: {raw_text!r}")

    # Stage 1: LM
    clean_text, lm_time = ask(raw_text, log)
    log.info(f"LM done in {lm_time:.2f}s → {clean_text!r}")

    # Stage 2: TTS
    log.info("loading Kokoro-82M …")
    pipeline = KPipeline(lang_code="f")
    log.info("model loaded")

    out_path = os.path.join(os.path.dirname(__file__), "output", "wire_out.wav")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    log.info(f"TTS input: {clean_text!r}")
    t0     = time.perf_counter()
    chunks = []
    for _, _, audio in pipeline(clean_text, voice="ff_siwis", speed=1.0):
        chunks.append(audio)
    tts_time = time.perf_counter() - t0

    samples   = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    audio_dur = len(samples) / 24000
    sf.write(out_path, samples, 24000)
    log.info(f"TTS done in {tts_time:.2f}s — audio: {audio_dur:.2f}s — RTF: {tts_time/max(audio_dur,0.01):.1f}x")
    log.info(f"saved → {out_path}")

    total = lm_time + tts_time
    log.info(f"=== wire total: {total:.2f}s  (LM {lm_time:.2f}s + TTS {tts_time:.2f}s) ===")

    subprocess.run(["afplay", out_path], check=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LM wrapper via Ollama")
    parser.add_argument("text", nargs="?", help="Raw input text")
    parser.add_argument("--test",  action="store_true", help="Run test inputs")
    parser.add_argument("--wire",  action="store_true", help="LM → TTS → play audio")
    parser.add_argument("--model", default=MODEL, choices=AVAILABLE_MODELS,
                        help=f"Ollama model (default: {MODEL})")
    args = parser.parse_args()

    if args.test:
        log = log_utils.setup("lm_test")
        log.info(f"model: {args.model}")
        for raw in TEST_INPUTS:
            text, elapsed = ask(raw, log, model=args.model)
            log.info(f"IN : {raw!r}")
            log.info(f"OUT: {text!r}  ({elapsed:.2f}s)")
        return

    if not args.text:
        parser.error("provide a text argument or use --test / --wire")

    if args.wire:
        wire(args.text)
    else:
        log = log_utils.setup("lm")
        text, elapsed = ask(args.text, log, model=args.model)
        print(text)


if __name__ == "__main__":
    main()
