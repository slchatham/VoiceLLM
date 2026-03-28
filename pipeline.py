"""
Full pipeline — mic → Parakeet → Qwen3.5 → Kokoro → speaker

All models loaded once at startup and kept in memory.

Usage:
    python pipeline.py
    python pipeline.py --no-play          # transcribe + LM only, no audio output
    python pipeline.py --model qwen3.5:9b
"""

import argparse
import os
import subprocess
import time

import numpy as np
import soundfile as sf

import log_utils
from stt import load_model as load_parakeet, transcribe, record_push_to_talk
from lm  import ask, AVAILABLE_MODELS, MODEL as DEFAULT_MODEL

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_WAV       = os.path.join(os.path.dirname(__file__), "output", "pipeline_out.wav")
SAMPLE_RATE      = 24000   # Kokoro output
CONTEXT_TIMEOUT  = 300     # seconds of silence before history reset
MAX_TURNS        = 5       # max exchanges kept in history (5 user + 5 assistant = 10 msgs)

VOICES = {"fr": "ff_siwis", "en": "af_heart"}

# French function words for language detection heuristic
_FR_WORDS = {
    "je", "tu", "il", "elle", "nous", "vous", "ils", "elles",
    "le", "la", "les", "un", "une", "des", "du", "de", "d",
    "et", "ou", "mais", "donc", "or", "ni", "car",
    "est", "sont", "a", "ont", "être", "avoir",
    "que", "qui", "quoi", "dont", "où",
    "ce", "se", "me", "te", "lui", "leur",
    "pas", "ne", "plus", "très", "bien", "tout",
    "en", "au", "aux", "sur", "sous", "dans", "avec", "pour", "par",
}

_EN_WORDS = {
    "i", "you", "he", "she", "we", "they",
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "and", "or", "but", "so", "yet", "nor",
    "it", "its", "this", "that", "these", "those",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "can", "could", "should", "may", "might",
    "not", "no", "very", "just", "all", "more",
    "in", "on", "at", "to", "of", "for", "with", "by", "from",
}


def _detect_lang(text: str) -> str:
    """Return 'fr' or 'en' based on function word frequency."""
    words = text.lower().split()
    fr_score = sum(1 for w in words if w.rstrip(".,!?;:") in _FR_WORDS)
    en_score = sum(1 for w in words if w.rstrip(".,!?;:") in _EN_WORDS)
    return "en" if en_score > fr_score else "fr"


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

def synthesize(pipelines: dict, text: str, log) -> tuple[float, float]:
    """Kokoro synthesis with automatic FR/EN routing. Returns (elapsed, audio_duration)."""
    lang     = _detect_lang(text)
    pipeline = pipelines[lang]
    voice    = VOICES[lang]
    log.info(f"TTS lang={lang} voice={voice}")

    t0     = time.perf_counter()
    chunks = []
    for _, _, audio in pipeline(text, voice=voice, speed=1.0):
        chunks.append(audio)
    elapsed = time.perf_counter() - t0

    samples   = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    audio_dur = len(samples) / SAMPLE_RATE
    sf.write(OUTPUT_WAV, samples, SAMPLE_RATE)
    log.info(f"TTS: {elapsed:.2f}s — audio: {audio_dur:.2f}s — RTF: {elapsed/max(audio_dur,0.01):.2f}x")
    return elapsed, audio_dur


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="VoiceLLM — fully local voice assistant (mic → STT → LM → TTS → speaker)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python pipeline.py                        default — qwen3.5:4b, ~4-6s round-trip
  python pipeline.py --model qwen3.5:9b     higher quality, ~10-25s round-trip
  python pipeline.py --think                enable reasoning mode (+10-30s, better accuracy)
  python pipeline.py --no-play              transcribe + LM only, no audio output
  python pipeline.py --think --no-play      reasoning mode without audio (fastest for testing)

push-to-talk:
  Enter     start recording
  Enter     stop recording and process
  Ctrl+C    quit
        """,
    )
    parser.add_argument("--no-play", action="store_true",
                        help="skip audio playback (TTS still runs, WAV saved to output/)")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=AVAILABLE_MODELS,
                        help=f"Ollama model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--think", action="store_true",
                        help="enable Qwen3.5 reasoning mode — slower but more accurate on factual queries")
    args = parser.parse_args()

    log = log_utils.setup("pipeline")
    os.makedirs(os.path.dirname(OUTPUT_WAV), exist_ok=True)

    # ── Banner ───────────────────────────────────────────────────────────────
    import datetime
    started = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    _W = 44
    _B = "\033[94m"
    _R = "\033[0m"
    def _row(s): return f"{_B}║{s:^{_W}}║{_R}"
    print(f"{_B}╔{'═'*_W}╗{_R}")
    print(_row("  V O I C E L L M   P I P E L I N E  "))
    _think_label = "  [think=ON]" if args.think else ""
    print(_row(f"Parakeet 0.6B · {args.model} · Kokoro-82M"))
    if args.think:
        print(_row(_think_label))
    print(_row(f"Started : {started}"))
    print(f"{_B}╚{'═'*_W}╝{_R}")
    print()

    # ── Load models ──────────────────────────────────────────────────────────
    log.info("=== VoiceLLM startup ===")

    log.info("[1/2] loading Parakeet TDT 0.6B-v3...")
    stt_model = load_parakeet(log)

    log.info("[2/2] loading Kokoro-82M (FR + EN, shared weights)...")
    from kokoro import KPipeline, KModel
    _kmodel = KModel()
    pipelines = {
        "fr": KPipeline(lang_code="f", model=_kmodel),
        "en": KPipeline(lang_code="a", model=_kmodel),
    }
    log.info("all models loaded — ready")

    # ── Push-to-talk loop ─────────────────────────────────────────────────────
    print("\nReady. Ctrl+C to quit.\n")

    history         = []
    last_turn_time  = None

    while True:
        try:
            # Stage 1 — STT
            audio = record_push_to_talk(log)
            if len(audio) < 16000 * 0.5:
                log.warning("recording too short, skipped")
                continue

            t_start = time.perf_counter()

            # Context timeout check
            now = time.perf_counter()
            if last_turn_time is not None and (now - last_turn_time) > CONTEXT_TIMEOUT:
                log.info(f"context reset (>{CONTEXT_TIMEOUT}s silence)")
                history.clear()

            # Sliding window — keep last MAX_TURNS exchanges
            if len(history) > MAX_TURNS * 2:
                history[:] = history[-(MAX_TURNS * 2):]

            raw_text, stt_time = transcribe(stt_model, audio, log)
            if not raw_text.strip():
                log.warning("empty transcription, skipped")
                continue

            # Stage 2 — LM (with context)
            clean_text, lm_time = ask(raw_text, log, history=history, model=args.model, think=args.think)
            if not clean_text.strip():
                log.warning("empty LM response, skipped")
                continue

            last_turn_time = time.perf_counter()

            # Stage 3 — TTS
            tts_time, audio_dur = synthesize(pipelines, clean_text, log)

            total = time.perf_counter() - t_start
            log.info(
                f"=== round-trip: {total:.2f}s  "
                f"(STT {stt_time:.2f}s + LM {lm_time:.2f}s + TTS {tts_time:.2f}s) "
                f"context={len(history)//2} turns ==="
            )

            # Stage 4 — play
            if not args.no_play:
                subprocess.run(["afplay", OUTPUT_WAV], check=True)

        except KeyboardInterrupt:
            log.info("interrupted — bye")
            print("\nBye.\n")
            os._exit(0)  # skip MPS/NeMo destructor — avoids bus error on macOS
        except Exception as exc:
            log.error(f"turn failed: {type(exc).__name__}: {exc}")
            print("\033[91m⚠  Error — retry possible.\033[0m", flush=True)


if __name__ == "__main__":
    main()
