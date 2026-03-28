"""
Phase 1 — Kokoro TTS wrapper
Kokoro-82M, French voice (ff_siwis)

Usage:
    python tts.py "Bonjour, comment ça va ?"
    python tts.py --test
    python tts.py --no-play
"""

import argparse
import os
import subprocess
import time

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VOICE       = "ff_siwis"   # French female (Kokoro built-in)
LANG_CODE   = "f"          # French
SAMPLE_RATE = 24000
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "output")

TEST_SENTENCES = [
    "Bonjour ! Je suis ton assistant vocal local. Comment puis-je t'aider ?",
    "Hello! This is a fully local voice assistant running on your Mac.",
    "Aujourd'hui on va tester le pipeline voix. Let's go !",
]

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    from kokoro import KPipeline
    return KPipeline(lang_code=LANG_CODE)


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def synthesize(pipeline, text: str, out_wav: str, speed: float = 1.0) -> tuple[float, float]:
    """Generate speech. Returns (elapsed_seconds, audio_duration_seconds)."""
    t0     = time.perf_counter()
    chunks = []
    for _, _, audio in pipeline(text, voice=VOICE, speed=speed):
        chunks.append(audio)
    elapsed = time.perf_counter() - t0

    samples   = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    audio_dur = len(samples) / SAMPLE_RATE
    sf.write(out_wav, samples, SAMPLE_RATE)
    print(f"[tts] {elapsed:.2f}s — audio: {audio_dur:.2f}s — RTF: {elapsed/max(audio_dur,0.01):.1f}x → {out_wav}")
    return elapsed, audio_dur


def play(wav_path: str):
    subprocess.run(["afplay", wav_path], check=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Kokoro TTS wrapper")
    parser.add_argument("text", nargs="?", help="Text to synthesize")
    parser.add_argument("--test", action="store_true", help="Run test sentences")
    parser.add_argument("--no-play", action="store_true", help="Skip audio playback")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[model] loading Kokoro-82M …")
    pipeline = load_model()
    print("[model] loaded")

    sentences = TEST_SENTENCES if args.test else [args.text]
    if not sentences[0]:
        parser.error("provide a text argument or use --test")

    for i, text in enumerate(sentences):
        print(f"\n[{i+1}/{len(sentences)}] {text!r}")
        wav     = os.path.join(OUTPUT_DIR, f"out_{i:02d}.wav")
        elapsed, audio_dur = synthesize(pipeline, text, wav)
        if not args.no_play:
            play(wav)


if __name__ == "__main__":
    main()
