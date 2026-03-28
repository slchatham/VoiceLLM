"""
Standalone benchmark: OuteTTS llama.cpp/Metal vs mlx-audio/MLX
Same sentences, same sampler params, same model (Llama-OuteTTS-1.0-1B).

No dependency on tts.py or any other project file.

NOTE — Decision record:
This benchmark was run to choose between two OuteTTS backends.
llama.cpp Q8_0 won on RTF (~3.1x vs ~3.7x for mlx-audio BF16).

However, OuteTTS was subsequently replaced by Kokoro-82M as the
production TTS for this pipeline. Reason: OuteTTS produced distorted
or broken audio ~50% of the time on M3 Pro (stochastic sampling
instability). Kokoro is deterministic, RTF 0.19–0.21x, and produces
consistently clean French output with the ff_siwis voice.

This file is kept as a record of the OuteTTS evaluation.

Usage:
    python bench_mlx.py
    python bench_mlx.py --no-play
"""

import argparse
import os
import subprocess
import time
import wave

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MLX_MODEL_ID  = "mlx-community/Llama-OuteTTS-1.0-1B-bf16"
OUT_DIR       = os.path.join(os.path.dirname(__file__), "output")

TEST_SENTENCES = [
    "Bonjour ! Je suis ton assistant vocal local. Comment puis-je t'aider ?",
    "Hello! This is a fully local voice assistant running on your Mac.",
    "Aujourd'hui on va tester le pipeline voix. Let's go !",
]

SAMPLER_KWARGS = dict(
    temperature=0.4,
    top_p=0.9,
    min_p=0.05,
    top_k=40,
    repetition_penalty=1.1,
    repetition_context_size=64,   # CRITICAL — 64-token window
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_wav(samples: np.ndarray, sr: int, path: str):
    pcm = (samples * 32767).astype(np.int16)
    with wave.open(path, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())


def play(path: str):
    subprocess.run(["afplay", path], check=True)


def fmt(v: float) -> str:
    return f"{v:.2f}s"


def print_table(rows: list[dict]):
    headers = ["backend", "sentence", "gen", "audio", "RTF"]
    col_w   = [10, 52, 7, 7, 6]

    def row_str(r):
        short = r["sentence"][:48] + "…" if len(r["sentence"]) > 48 else r["sentence"]
        return [r["backend"], short, fmt(r["gen"]), fmt(r["audio"]), f"{r['rtf']:.1f}x"]

    sep = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"
    hdr = "|" + "|".join(f" {h:<{w}} " for h, w in zip(headers, col_w)) + "|"
    print(sep)
    print(hdr)
    print(sep)
    for r in rows:
        cells = row_str(r)
        print("|" + "|".join(f" {c:<{w}} " for c, w in zip(cells, col_w)) + "|")
    print(sep)

    # Summary: mean RTF per backend
    backends = sorted({r["backend"] for r in rows})
    print("\nMean RTF:")
    for b in backends:
        rtfs = [r["rtf"] for r in rows if r["backend"] == b]
        print(f"  {b}: {sum(rtfs)/len(rtfs):.2f}x")


# ---------------------------------------------------------------------------
# Backend A — llama.cpp via outetts lib
# ---------------------------------------------------------------------------

def run_llamacpp(sentences: list[str], play_audio: bool) -> list[dict]:
    import outetts

    config = outetts.ModelConfig.auto_config(
        model=outetts.Models.VERSION_1_0_SIZE_1B,
        backend=outetts.Backend.LLAMACPP,
        quantization=outetts.LlamaCppQuantization.Q8_0,
    )
    tts   = outetts.Interface(config)
    spk   = tts.load_default_speaker("EN-FEMALE-1-NEUTRAL")
    smpl  = outetts.SamplerConfig(
        temperature=SAMPLER_KWARGS["temperature"],
        repetition_penalty=SAMPLER_KWARGS["repetition_penalty"],
        repetition_range=SAMPLER_KWARGS["repetition_context_size"],
        top_k=SAMPLER_KWARGS["top_k"],
        top_p=SAMPLER_KWARGS["top_p"],
        min_p=SAMPLER_KWARGS["min_p"],
    )

    results = []
    for i, text in enumerate(sentences):
        gen_cfg = outetts.GenerationConfig(text=text, speaker=spk, sampler_config=smpl)
        t0  = time.perf_counter()
        out = tts.generate(gen_cfg)
        gen = time.perf_counter() - t0

        samples   = out.audio.squeeze().numpy() if hasattr(out.audio, "numpy") else np.array(out.audio).squeeze()
        audio_dur = samples.shape[-1] / out.sr
        wav_path  = os.path.join(OUT_DIR, f"bench_llamacpp_{i:02d}.wav")
        save_wav(samples, out.sr, wav_path)

        if play_audio:
            play(wav_path)

        results.append({"backend": "llamacpp", "sentence": text, "gen": gen, "audio": audio_dur, "rtf": gen / audio_dur})

    return results


# ---------------------------------------------------------------------------
# Backend B — mlx-audio
# ---------------------------------------------------------------------------

def run_mlx(sentences: list[str], play_audio: bool) -> list[dict]:
    from mlx_audio.tts import load
    import mlx.core as mx

    print(f"[mlx] loading {MLX_MODEL_ID} …")
    model = load(MLX_MODEL_ID)

    results = []
    for i, text in enumerate(sentences):
        t0         = time.perf_counter()
        chunks     = []
        sr         = 24000
        for chunk in model.generate(text, **SAMPLER_KWARGS):
            audio_np = np.array(chunk.audio).squeeze()
            chunks.append(audio_np)
            sr = chunk.sample_rate
        gen     = time.perf_counter() - t0

        samples   = np.concatenate(chunks) if chunks else np.array([])
        audio_dur = samples.shape[-1] / sr if samples.ndim > 0 and samples.shape[-1] > 0 else 1.0
        wav_path  = os.path.join(OUT_DIR, f"bench_mlx_{i:02d}.wav")
        save_wav(samples, sr, wav_path)

        if play_audio:
            play(wav_path)

        results.append({"backend": "mlx-audio", "sentence": text, "gen": gen, "audio": audio_dur, "rtf": gen / audio_dur})

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-play", action="store_true")
    parser.add_argument("--backend", choices=["llamacpp", "mlx", "both"], default="both")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    play_audio = not args.no_play
    all_rows   = []

    if args.backend in ("llamacpp", "both"):
        print("\n=== llama.cpp backend ===")
        all_rows += run_llamacpp(TEST_SENTENCES, play_audio)

    if args.backend in ("mlx", "both"):
        print(f"\n=== mlx-audio backend ({MLX_MODEL_ID}) ===")
        try:
            all_rows += run_mlx(TEST_SENTENCES, play_audio)
        except Exception as e:
            print(f"[mlx] FAILED: {e}")
            print(f"[mlx] Hint: check that {MLX_MODEL_ID} exists on HuggingFace.")

    if all_rows:
        print("\n=== Benchmark Results ===")
        print_table(all_rows)


if __name__ == "__main__":
    main()
