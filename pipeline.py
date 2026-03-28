"""
Full pipeline — mic → Parakeet → Qwen3.5 → Kokoro → speaker

All models loaded once at startup and kept in memory.

Usage:
    python pipeline.py
    python pipeline.py --no-play   # transcribe + LM only, no audio output
"""

import argparse
import os
import subprocess
import time

import numpy as np
import soundfile as sf

import log_utils
from stt import load_model as load_parakeet, transcribe, record_push_to_talk
from lm  import ask

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_WAV       = os.path.join(os.path.dirname(__file__), "output", "pipeline_out.wav")
SAMPLE_RATE      = 24000   # Kokoro output
CONTEXT_TIMEOUT  = 300     # seconds of silence before history reset
MAX_TURNS        = 5       # max exchanges kept in history (5 user + 5 assistant = 10 msgs)


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

def synthesize(pipeline, text: str, log) -> tuple[float, float]:
    """Kokoro synthesis. Returns (elapsed, audio_duration)."""
    t0     = time.perf_counter()
    chunks = []
    for _, _, audio in pipeline(text, voice="ff_siwis", speed=1.0):
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
    parser = argparse.ArgumentParser(description="VoiceLLM pipeline")
    parser.add_argument("--no-play", action="store_true", help="Skip audio playback")
    args = parser.parse_args()

    log = log_utils.setup("pipeline")
    os.makedirs(os.path.dirname(OUTPUT_WAV), exist_ok=True)

    # ── Load models ──────────────────────────────────────────────────────────
    log.info("=== VoiceLLM startup ===")

    log.info("[1/2] loading Parakeet TDT 0.6B-v3 …")
    stt_model = load_parakeet(log)

    log.info("[2/2] loading Kokoro-82M …")
    from kokoro import KPipeline
    tts_pipeline = KPipeline(lang_code="f")
    log.info("all models loaded — ready")

    # ── Push-to-talk loop ─────────────────────────────────────────────────────
    print("\nVoiceLLM ready. Ctrl+C to quit.\n")

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
            clean_text, lm_time = ask(raw_text, log, history=history)
            if not clean_text.strip():
                log.warning("empty LM response, skipped")
                continue

            last_turn_time = time.perf_counter()

            # Stage 3 — TTS
            tts_time, audio_dur = synthesize(tts_pipeline, clean_text, log)

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
            print("\nBye.")
            break


if __name__ == "__main__":
    main()
